from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# int16 range
_I16_MAX = np.float32(32767.0)
_I16_MIN = np.float32(-32768.0)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 1600  # 100ms chunks at 16kHz
    device: Optional[int] = None  # None = system default mic


class AudioCapture:
    """Captures audio from the microphone with software gain / AGC and pushes PCM bytes to consumers."""

    def __init__(self, config: AudioConfig | None = None):
        self._config = config or AudioConfig()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording: bool = False
        self._chunks_sent: int = 0
        # Direct sync callback for audio data (called from audio thread)
        self._on_audio: Optional[Callable[[bytes], None]] = None
        # Callback for reporting audio levels (called from audio thread)
        self._on_level: Optional[Callable[[float, float], None]] = None

        # --- Gain / AGC state ---
        self._gain: float = 3.0          # manual gain multiplier (default 3x boost)
        self._agc_enabled: bool = True   # automatic gain control on/off
        self._agc_target: float = 8000.0  # target RMS level for AGC (~25% of int16 range)
        self._agc_gain: float = 1.0      # current AGC computed gain
        self._agc_attack: float = 0.05   # how fast AGC increases gain (per chunk)
        self._agc_release: float = 0.02  # how fast AGC decreases gain (per chunk)
        self._agc_max: float = 20.0      # maximum AGC gain
        self._agc_min: float = 1.0       # minimum AGC gain

    def set_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set a sync callback that receives PCM bytes directly from the audio thread."""
        self._on_audio = callback

    def set_level_callback(self, callback: Callable[[float, float], None]) -> None:
        """Set a callback that receives (rms_normalized, peak_normalized) for UI meters."""
        self._on_level = callback

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(0.5, min(value, 20.0))
        logger.info(f"Manual gain set to {self._gain:.1f}x")

    @property
    def agc_enabled(self) -> bool:
        return self._agc_enabled

    @agc_enabled.setter
    def agc_enabled(self, value: bool) -> None:
        self._agc_enabled = value
        if not value:
            self._agc_gain = 1.0
        logger.info(f"AGC {'enabled' if value else 'disabled'}")

    @property
    def effective_gain(self) -> float:
        """Total gain applied = manual * AGC."""
        return self._gain * self._agc_gain

    def _apply_gain(self, samples: np.ndarray) -> np.ndarray:
        """Apply manual gain + AGC to int16 samples and return amplified int16 array."""
        # Convert to float32 for processing
        audio_f = samples.astype(np.float32)

        # Update AGC gain based on current signal level
        if self._agc_enabled:
            rms = np.sqrt(np.mean(audio_f ** 2)) + 1e-10
            current_rms = rms * self._gain * self._agc_gain
            if current_rms < self._agc_target:
                # Signal too quiet — increase gain
                self._agc_gain += self._agc_attack
            elif current_rms > self._agc_target * 1.5:
                # Signal too loud — decrease gain
                self._agc_gain -= self._agc_release
            self._agc_gain = max(self._agc_min, min(self._agc_gain, self._agc_max))

        # Apply total gain
        total_gain = self._gain * self._agc_gain
        audio_f *= np.float32(total_gain)

        # Clip to int16 range and convert back
        np.clip(audio_f, _I16_MIN, _I16_MAX, out=audio_f)
        return audio_f.astype(np.int16)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning(f"Audio status: {status}")
        if self._is_recording and self._on_audio:
            # Apply gain/AGC
            amplified = self._apply_gain(indata.copy())
            pcm_bytes = amplified.tobytes()
            self._on_audio(pcm_bytes)
            self._chunks_sent += 1

            # Report audio levels every 10 chunks (~2.5s)
            if self._chunks_sent % 10 == 1:
                flat = amplified.flatten().astype(np.float32)
                rms = float(np.sqrt(np.mean(flat ** 2)))
                peak = float(np.max(np.abs(flat)))
                rms_norm = min(rms / _I16_MAX, 1.0)
                peak_norm = min(peak / _I16_MAX, 1.0)
                if self._on_level:
                    self._on_level(rms_norm, peak_norm)

            if self._chunks_sent % 100 == 1:
                flat = amplified.flatten().astype(np.float32)
                peak = float(np.max(np.abs(flat)))
                logger.info(
                    f"Mic: chunk #{self._chunks_sent}, peak={peak:.0f}, "
                    f"gain={self._gain:.1f}x, agc={self._agc_gain:.1f}x, "
                    f"total={self.effective_gain:.1f}x"
                )
        elif not self._on_audio:
            if not hasattr(self, '_warned_no_cb'):
                logger.warning("Audio callback fired but no _on_audio consumer set!")
                self._warned_no_cb = True

    @staticmethod
    def _find_builtin_mic() -> Optional[int]:
        """Find the best available microphone device index."""
        devices = sd.query_devices()
        # Priority 1: built-in mic keywords
        builtin_keywords = ["macbook", "built-in", "встроенный", "internal", "mikrofon", "microphone"]
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                name = d["name"].lower()
                if any(kw in name for kw in builtin_keywords):
                    return i
        # Priority 2: first available input device
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                return i
        return None

    async def start(self) -> None:
        self._is_recording = True
        self._agc_gain = 1.0  # reset AGC state

        # Maximize system input volume on macOS
        self._set_system_input_volume(100)

        # Always use built-in MacBook mic
        device = self._find_builtin_mic()
        if device is not None:
            logger.info(f"Using built-in mic: device #{device}")
        else:
            device = self._config.device
            logger.warning("Built-in mic not found, using default device")

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self._config.blocksize,
            device=device,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(
            f"Audio capture started: {self._config.sample_rate}Hz, "
            f"blocksize={self._config.blocksize}, gain={self._gain}x, agc={self._agc_enabled}"
        )

    @staticmethod
    def _set_system_input_volume(volume: int) -> None:
        """Set macOS system microphone input volume (0-100)."""
        import subprocess
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume input volume {volume}"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            logger.info(f"System input volume set to {volume}%")
        except Exception as e:
            logger.warning(f"Could not set system input volume: {e}")

    async def stop(self) -> None:
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped")

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @staticmethod
    def list_devices() -> list[dict]:
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
