import asyncio
import logging
import time
from typing import Optional

from backend.correction.ollama_corrector import OllamaCorrector
from backend.correction.gemini_corrector import GeminiCorrector
from backend.models.correction import CorrectionResult
from backend.models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

# Accumulate segments until this many words before sending to correction
BATCH_WORD_TARGET = 200

# Max time to wait for more segments before flushing the batch (seconds)
BATCH_TIMEOUT_SEC = 15.0

# How many previous corrected blocks to keep as context
MAX_CONTEXT_BLOCKS = 10


class CorrectionManager:
    """
    Batches transcript segments (~200 words) and sends combined text
    to AI for correction with full history of previously corrected blocks.
    """

    def __init__(
        self,
        ollama: OllamaCorrector,
        gemini: Optional[GeminiCorrector] = None,
    ):
        self._ollama = ollama
        self._gemini = gemini
        self._segment_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._result_queue: asyncio.Queue[CorrectionResult] = asyncio.Queue()
        self._running = False
        self._batcher_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._batch_queue: asyncio.Queue[list[TranscriptSegment]] = asyncio.Queue()
        # History of corrected blocks for context
        self._corrected_history: list[str] = []
        # Cache for ollama availability (60s TTL, invalidated on failure)
        self._ollama_available: Optional[bool] = None
        self._ollama_checked_at: float = 0.0

    async def _check_ollama(self) -> bool:
        """Check Ollama availability with 60s cache to avoid per-batch HTTP overhead."""
        now = time.monotonic()
        if self._ollama_available is not None and (now - self._ollama_checked_at) < 60.0:
            return self._ollama_available
        self._ollama_available = await self._ollama.is_available()
        self._ollama_checked_at = now
        return self._ollama_available

    async def start(self) -> None:
        self._running = True
        self._corrected_history = []
        self._ollama_available = None
        self._ollama_checked_at = 0.0
        self._batcher_task = asyncio.create_task(self._batcher())
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(f"Correction manager started (batch target: {BATCH_WORD_TARGET} words)")

    async def stop(self) -> None:
        self._running = False
        for task in [self._batcher_task, self._worker_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._batcher_task = None
        self._worker_task = None
        logger.info("Correction manager stopped")

    async def submit(self, segment: TranscriptSegment, context: str = "") -> None:
        await self._segment_queue.put(segment)

    async def get_result(self) -> CorrectionResult:
        return await self._result_queue.get()

    def has_results(self) -> bool:
        return not self._result_queue.empty()

    async def _batcher(self) -> None:
        """Accumulates segments into batches of ~200 words."""
        batch: list[TranscriptSegment] = []
        word_count = 0
        last_segment_time = time.monotonic()

        while self._running:
            timeout = BATCH_TIMEOUT_SEC
            if batch:
                elapsed = time.monotonic() - last_segment_time
                timeout = max(0.1, BATCH_TIMEOUT_SEC - elapsed)

            try:
                segment = await asyncio.wait_for(
                    self._segment_queue.get(), timeout=timeout
                )
                batch.append(segment)
                word_count += len(segment.text.split())
                last_segment_time = time.monotonic()

                if word_count >= BATCH_WORD_TARGET:
                    await self._flush_batch(batch)
                    batch = []
                    word_count = 0

            except asyncio.TimeoutError:
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                    word_count = 0
            except asyncio.CancelledError:
                if batch:
                    await self._flush_batch(batch)
                break

    async def _flush_batch(self, batch: list[TranscriptSegment]) -> None:
        total_words = sum(len(s.text.split()) for s in batch)
        logger.info(f"Flushing batch: {len(batch)} segments, {total_words} words")
        await self._batch_queue.put(list(batch))

    def _build_context(self) -> str:
        """Build context string from previously corrected blocks."""
        if not self._corrected_history:
            return ""
        return "\n\n".join(self._corrected_history)

    async def _worker(self) -> None:
        """Processes batched segments through AI correction."""
        while self._running:
            try:
                batch = await asyncio.wait_for(
                    self._batch_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Process remaining batches before exiting
                while not self._batch_queue.empty():
                    try:
                        batch = self._batch_queue.get_nowait()
                        context = self._build_context()
                        result = await self._correct_batch(batch, context)
                        if result:
                            await self._result_queue.put(result)
                    except Exception:
                        break
                break

            start = time.monotonic()
            context = self._build_context()
            result = await self._correct_batch(batch, context)
            elapsed = (time.monotonic() - start) * 1000

            if result:
                # Save corrected text to history for future context
                if result.corrected_text:
                    self._corrected_history.append(result.corrected_text)
                    # Keep only last N blocks
                    if len(self._corrected_history) > MAX_CONTEXT_BLOCKS:
                        self._corrected_history = self._corrected_history[-MAX_CONTEXT_BLOCKS:]

                logger.info(
                    f"Batch corrected in {elapsed:.0f}ms "
                    f"({len(result.items)} corrections, "
                    f"{len(batch)} segments merged, "
                    f"context: {len(self._corrected_history)} blocks)"
                )
                await self._result_queue.put(result)

    async def _correct_batch(
        self, batch: list[TranscriptSegment], context: str
    ) -> Optional[CorrectionResult]:
        """Combine segments and send to AI for correction with context."""
        combined_text = " ".join(s.text for s in batch)
        segment_ids = [s.segment_id for s in batch]

        fake_segment = TranscriptSegment(
            segment_id=segment_ids[0],
            speaker=batch[0].speaker,
            speaker_role=batch[0].speaker_role,
            text=combined_text,
            words=[],
            timestamp=batch[0].timestamp,
            is_final=True,
        )

        # Try Ollama first (with cached availability check)
        try:
            if await self._check_ollama():
                result = await self._ollama.correct(fake_segment, context)
                result.segment_id = "|".join(segment_ids)
                return result
        except Exception as e:
            logger.warning(f"Ollama batch correction failed: {e}")
            # Invalidate cache on failure so next batch re-checks
            self._ollama_available = None

        # Fallback to Gemini
        if self._gemini:
            try:
                result = await self._gemini.correct(fake_segment, context)
                result.segment_id = "|".join(segment_ids)
                return result
            except Exception as e:
                logger.error(f"Gemini batch correction also failed: {e}")

        # Both failed
        logger.warning("All correctors unavailable, passing through")
        return CorrectionResult(
            segment_id="|".join(segment_ids),
            corrected_text=combined_text,
            items=[],
            model_used="none",
            processing_time_ms=0,
            confidence=0.0,
        )
