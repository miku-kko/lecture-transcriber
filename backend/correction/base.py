from abc import ABC, abstractmethod

from backend.models.correction import CorrectionResult
from backend.models.transcript import TranscriptSegment


class BaseCorrector(ABC):
    @abstractmethod
    async def correct(
        self, segment: TranscriptSegment, context: str = ""
    ) -> CorrectionResult:
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        ...
