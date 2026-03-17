import json
import logging
import time

import aiohttp

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
      "original": "bledny fragment",
      "suggested": "poprawny fragment",
      "type": "grammar",
      "severity": "warning",
      "explanation": "wyjasnienie po polsku"
    }
  ],
  "confidence": 0.95
}

WAZNE: Pole "type" musi byc DOKLADNIE jedna z wartosci: grammar, spelling, logic, punctuation, style.
Pole "severity" musi byc DOKLADNIE jedna z wartosci: info, warning, error.

Jesli tekst jest poprawny, zwroc go bez zmian z pusta lista corrections."""


class OllamaCorrector(BaseCorrector):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "SpeakLeash/bielik-7b-instruct-v0.1-gguf",
        timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def correct(
        self, segment: TranscriptSegment, context: str = ""
    ) -> CorrectionResult:
        start_time = time.monotonic()

        context_section = ""
        if context:
            context_section = f"WCZESNIEJ SKORYGOWANY TEKST WYKLADU (uzyj jako kontekst tematyczny, terminologiczny i stylistyczny):\n---\n{context}\n---\n\n"

        user_prompt = f"{context_section}NOWY FRAGMENT DO KOREKTY:\n\"{segment.text}\"\n\nMowca: {segment.speaker_role.value}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_PL},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
                "top_p": 0.9,
            },
        }

        session = await self._get_session()
        async with session.post(
            f"{self._base_url}/api/chat", json=payload
        ) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Ollama returned {resp.status}")
            data = await resp.json()

        elapsed_ms = (time.monotonic() - start_time) * 1000
        response_text = data["message"]["content"]
        return self._parse_response(response_text, segment.segment_id, elapsed_ms)

    def _parse_response(
        self, raw: str, segment_id: str, elapsed_ms: float
    ) -> CorrectionResult:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Ollama JSON response: {raw[:200]}")
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
                items.append(
                    CorrectionItem(
                        original_text=c.get("original", ""),
                        suggested_text=c.get("suggested", ""),
                        correction_type=CorrectionType(c.get("type", "grammar")),
                        severity=CorrectionSeverity(c.get("severity", "info")),
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
        self._compute_offsets(result)
        return result

    def _compute_offsets(self, result: CorrectionResult) -> None:
        for item in result.items:
            idx = result.corrected_text.find(item.suggested_text)
            if idx >= 0:
                item.char_start = idx
                item.char_end = idx + len(item.suggested_text)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_available(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self._base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return any(self._model in m for m in models)
            return False
        except Exception:
            return False
