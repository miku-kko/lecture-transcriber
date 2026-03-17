import asyncio
import json
import logging
import time
from collections import deque

from backend.correction.base import BaseCorrector
from backend.models.correction import (
    CorrectionItem,
    CorrectionResult,
    CorrectionSeverity,
    CorrectionType,
)
from backend.models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PL = """Jestes asystentem korektora tekstu transkrypcji wykladow akademickich w jezyku polskim.

Otrzymujesz DLUZSZY fragment transkrypcji (speech-to-text, ok. 200 slow) zlozony z wielu krotkich segmentow.
Twoje zadanie:
1. POLACZ wszystkie fragmenty w jeden spojny, czytelny tekst
2. Popraw bledy ortograficzne, gramatyczne i interpunkcyjne
3. Usun powtorzenia, slowa-wypelniacze (np. "eee", "no", "znaczy")
4. Popraw logike i spojnosc zdan
5. Zachowaj oryginalny sens i styl akademicki wypowiedzi
6. W polu "corrections" wymien NAJWAZNIEJSZE zmiany (max 10)

Odpowiedz WYLACZNIE w formacie JSON:
{
  "corrected_text": "pelny poprawiony i polaczony tekst",
  "corrections": [
    {
      "original": "bledny lub niejasny fragment z oryginalu",
      "suggested": "poprawiona wersja",
      "type": "grammar",
      "severity": "warning",
      "explanation": "krotkie wyjasnienie po polsku"
    }
  ],
  "confidence": 0.95
}

WAZNE: Pole "type" musi byc DOKLADNIE jedna z wartosci: grammar, spelling, logic, punctuation, style.
Pole "severity" musi byc DOKLADNIE jedna z wartosci: info, warning, error.

Jesli tekst jest poprawny, zwroc go bez zmian z pusta lista corrections."""


class GeminiCorrector(BaseCorrector):
    # Max requests per minute (free tier = 15 RPM for 2.0-flash)
    MAX_RPM = 12

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._request_times: deque = deque()
        self._rate_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self) -> None:
        """Wait if we're hitting the rate limit."""
        async with self._rate_lock:
            now = time.monotonic()
            # Remove requests older than 60 seconds
            while self._request_times and now - self._request_times[0] > 60:
                self._request_times.popleft()
            # If at limit, wait until oldest request expires
            if len(self._request_times) >= self.MAX_RPM:
                wait_time = 60 - (now - self._request_times[0]) + 0.5
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
            self._request_times.append(time.monotonic())

    async def correct(
        self, segment: TranscriptSegment, context: str = ""
    ) -> CorrectionResult:
        await self._wait_for_rate_limit()
        start_time = time.monotonic()

        context_section = ""
        if context:
            context_section = f"WCZESNIEJ SKORYGOWANY TEKST WYKLADU (uzyj jako kontekst tematyczny, terminologiczny i stylistyczny):\n---\n{context}\n---\n\n"

        user_prompt = f"{context_section}NOWY FRAGMENT DO KOREKTY:\n\"{segment.text}\"\n\nMowca: {segment.speaker_role.value}"

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT_PL}\n\n{user_prompt}"}]}
                    ],
                    config=self._genai.types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                ),
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return self._parse_response(response.text, segment.segment_id, elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Gemini correction failed: {e}")
            return CorrectionResult(
                segment_id=segment.segment_id,
                corrected_text=segment.text,
                items=[],
                model_used=self._model,
                processing_time_ms=elapsed_ms,
                confidence=0.0,
            )

    def _parse_response(
        self, raw: str, segment_id: str, elapsed_ms: float
    ) -> CorrectionResult:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Gemini JSON: {raw[:200]}")
            return CorrectionResult(
                segment_id=segment_id,
                corrected_text="",
                items=[],
                model_used=self._model,
                processing_time_ms=elapsed_ms,
                confidence=0.0,
            )

        items = []
        for c in parsed.get("corrections", []):
            try:
                raw_type = c.get("type", "grammar")
                # Handle compound types like "grammar|style" — take the first one
                if "|" in raw_type:
                    raw_type = raw_type.split("|")[0].strip()
                try:
                    ctype = CorrectionType(raw_type)
                except ValueError:
                    ctype = CorrectionType.GRAMMAR

                raw_severity = c.get("severity", "info")
                if "|" in raw_severity:
                    raw_severity = raw_severity.split("|")[0].strip()
                try:
                    cseverity = CorrectionSeverity(raw_severity)
                except ValueError:
                    cseverity = CorrectionSeverity.INFO

                items.append(
                    CorrectionItem(
                        original_text=c.get("original", ""),
                        suggested_text=c.get("suggested", ""),
                        correction_type=ctype,
                        severity=cseverity,
                        explanation=c.get("explanation", ""),
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed correction: {e}")

        result = CorrectionResult(
            segment_id=segment_id,
            corrected_text=parsed.get("corrected_text", ""),
            items=items,
            model_used=self._model,
            processing_time_ms=elapsed_ms,
            confidence=parsed.get("confidence", 0.5),
        )
        return result

    async def is_available(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._client.models.list)
            return True
        except Exception:
            return False
