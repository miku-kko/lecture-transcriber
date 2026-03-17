from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Optional

import websockets
import websockets.sync.client

from backend.config import Settings
from backend.models.transcript import SpeakerRole, TranscriptSegment, Word

logger = logging.getLogger(__name__)

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTTClient:
    """
    Manages a raw WebSocket connection to Deepgram for real-time Polish STT.

    Uses websockets library directly for full control over timing and keepalive.
    Runs in a background thread with asyncio queue bridge.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._segment_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=500)
        self._is_connected: bool = False
        self._speaker_map: dict[int, SpeakerRole] = {0: SpeakerRole.LECTURER}
        self._recv_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._ws_lock = threading.Lock()
        self._keyterms: list[str] = []

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()

        # Drain any old audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._recv_thread = threading.Thread(target=self._run_connection, daemon=True)
        self._recv_thread.start()

        # Wait for connection to be established
        for _ in range(40):
            if self._is_connected:
                break
            await asyncio.sleep(0.1)

        if self._is_connected:
            logger.info("Deepgram streaming connected")
        else:
            logger.error("Deepgram connection timed out")

    def set_keyterms(self, terms: list[str]) -> None:
        """Set keyterms for boosted recognition (up to 100)."""
        self._keyterms = terms[:100]

    def _build_url(self) -> str:
        params = {
            "model": "nova-3",
            "language": "pl",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "punctuate": "true",
            "diarize": "true",
            "smart_format": "true",
            "interim_results": "true",
            "utterance_end_ms": "1000",
            "vad_events": "true",
            "endpointing": "150",
            "profanity_filter": "false",
            "numerals": "true",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        # Keyterms are added as repeated params
        for term in self._keyterms:
            from urllib.parse import quote
            query += f"&keywords={quote(term)}"
        return f"{DEEPGRAM_WS_URL}?{query}"

    _MAX_RECONNECT_ATTEMPTS = 3
    _RECONNECT_DELAY_SEC = 2.0

    def _run_connection(self) -> None:
        """Open WebSocket and run recv loop in this thread, with auto-reconnect."""
        attempt = 0
        while not self._stop_event.is_set():
            url = self._build_url()
            headers = {"Authorization": f"Token {self._settings.deepgram_api_key}"}

            try:
                self._ws = websockets.sync.client.connect(
                    url,
                    additional_headers=headers,
                    close_timeout=5,
                )
                self._is_connected = True
                attempt = 0  # Reset on successful connection
                logger.info("Deepgram WebSocket opened")

                # Start sender in separate thread
                self._send_thread = threading.Thread(target=self._send_audio_loop, daemon=True)
                self._send_thread.start()

                # Recv loop (blocks until connection closes)
                while not self._stop_event.is_set():
                    try:
                        raw = self._ws.recv(timeout=1.0)
                        if isinstance(raw, str):
                            self._handle_message(raw)
                    except TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Deepgram connection closed: {e}")
                        break
                    except Exception as e:
                        if not self._stop_event.is_set():
                            logger.error(f"Deepgram recv error: {e}")
                        break

            except Exception as e:
                logger.error(f"Deepgram connection error: {e}")
            finally:
                self._is_connected = False

            # Auto-reconnect logic
            if self._stop_event.is_set():
                break
            attempt += 1
            if attempt > self._MAX_RECONNECT_ATTEMPTS:
                logger.error(f"Deepgram: max reconnect attempts ({self._MAX_RECONNECT_ATTEMPTS}) reached, giving up")
                break
            logger.info(f"Deepgram: reconnecting (attempt {attempt}/{self._MAX_RECONNECT_ATTEMPTS}) in {self._RECONNECT_DELAY_SEC}s...")
            self._stop_event.wait(self._RECONNECT_DELAY_SEC)

    def _send_audio_loop(self) -> None:
        """Send audio from the queue to Deepgram, with keepalive silence."""
        silence = b"\x00" * 8000
        last_send = time.time()
        chunks_sent = 0

        logger.info("Send audio loop started")

        while not self._stop_event.is_set():
            try:
                audio = self._audio_queue.get(timeout=0.5)
                with self._ws_lock:
                    ws = self._ws
                    if not ws or not self._is_connected:
                        continue
                    ws.send(audio)
                last_send = time.time()
                chunks_sent += 1
                if chunks_sent % 40 == 1:
                    logger.info(f"Audio sent: {chunks_sent} chunks, {len(audio)} bytes")
            except queue.Empty:
                # Send silence keepalive every 5 seconds
                if time.time() - last_send > 5.0:
                    try:
                        with self._ws_lock:
                            ws = self._ws
                            if ws and self._is_connected:
                                ws.send(silence)
                        last_send = time.time()
                    except Exception as e:
                        logger.warning(f"Keepalive send failed: {e}")
                        break
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Send audio error: {type(e).__name__}: {e}")
                break

        logger.info(f"Send audio loop ended after {chunks_sent} chunks")

    def _handle_message(self, raw: str) -> None:
        """Parse and handle a message from Deepgram."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "Results":
            self._handle_transcript(msg)
        elif msg_type == "UtteranceEnd":
            logger.info("Utterance end")
        elif msg_type == "SpeechStarted":
            logger.info("Speech started")
        elif msg_type == "Metadata":
            logger.info(f"Deepgram metadata: request_id={msg.get('request_id', '?')}")
        else:
            logger.info(f"Deepgram message type: {msg_type}")

    def _handle_transcript(self, msg: dict) -> None:
        """Process transcript result and push segments to the async queue."""
        try:
            channel = msg.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                return

            alt = alternatives[0]
            transcript = alt.get("transcript", "").strip()
            if not transcript:
                return

            is_final = msg.get("is_final", False)

            words = []
            for w in alt.get("words", []):
                speaker = w.get("speaker", 0) or 0
                words.append(
                    Word(
                        text=w.get("word", ""),
                        speaker=speaker,
                        confidence=w.get("confidence", 0.0),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                    )
                )

            if not words:
                words = [
                    Word(text=transcript, speaker=0, confidence=alt.get("confidence", 0.0))
                ]

            segments = self._group_words_by_speaker(words, is_final)
            for seg in segments:
                if self._loop:
                    self._loop.call_soon_threadsafe(
                        self._segment_queue.put_nowait, seg
                    )
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")

    def _group_words_by_speaker(
        self, words: list[Word], is_final: bool
    ) -> list[TranscriptSegment]:
        if not words:
            return []

        segments = []
        current_speaker = words[0].speaker
        current_words = [words[0]]

        for word in words[1:]:
            if word.speaker != current_speaker:
                segments.append(
                    self._make_segment(current_words, current_speaker, is_final)
                )
                current_speaker = word.speaker
                current_words = [word]
            else:
                current_words.append(word)

        segments.append(self._make_segment(current_words, current_speaker, is_final))
        return segments

    def _make_segment(
        self, words: list[Word], speaker: int, is_final: bool
    ) -> TranscriptSegment:
        role = self._speaker_map.get(speaker, SpeakerRole.UNKNOWN)
        return TranscriptSegment(
            segment_id=str(uuid.uuid4()),
            speaker=speaker,
            speaker_role=role,
            text=" ".join(w.text for w in words),
            words=words,
            timestamp=time.time(),
            is_final=is_final,
        )

    def send_audio_sync(self, audio_bytes: bytes) -> None:
        """Queue audio to be sent to Deepgram (called from any thread)."""
        if not self._is_connected:
            return
        try:
            self._audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            # Drop oldest chunk to make room (prevent permanent queue stall)
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Queue audio to be sent to Deepgram (called from async code)."""
        self.send_audio_sync(audio_bytes)

    async def disconnect(self) -> None:
        self._stop_event.set()
        self._is_connected = False
        with self._ws_lock:
            if self._ws:
                try:
                    # Send close frame to Deepgram
                    self._ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
        if self._send_thread:
            self._send_thread.join(timeout=3)
            self._send_thread = None
        if self._recv_thread:
            self._recv_thread.join(timeout=3)
            self._recv_thread = None
        # Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Deepgram disconnected")

    async def get_segment(self) -> TranscriptSegment:
        return await self._segment_queue.get()

    def set_speaker_map(self, mapping: dict[int, SpeakerRole]) -> None:
        self._speaker_map = mapping

    @property
    def is_connected(self) -> bool:
        return self._is_connected
