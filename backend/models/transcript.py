from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from backend.models.correction import CorrectionResult


class SpeakerRole(Enum):
    LECTURER = "lecturer"
    STUDENT = "student"
    UNKNOWN = "unknown"


@dataclass
class Word:
    text: str
    speaker: int
    confidence: float
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class TranscriptSegment:
    segment_id: str
    speaker: int
    speaker_role: SpeakerRole
    text: str
    words: list[Word]
    timestamp: float
    is_final: bool = False
    is_corrected: bool = False
    correction: Any = None  # CorrectionResult | None

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role.value,
            "text": self.text,
            "is_final": self.is_final,
            "is_corrected": self.is_corrected,
            "timestamp": self.timestamp,
        }


@dataclass
class TranscriptSession:
    session_id: str
    title: str
    started_at: float
    ended_at: Optional[float] = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    speaker_map: dict[int, SpeakerRole] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def create(title: str = "", metadata: dict | None = None) -> TranscriptSession:
        if not title:
            title = f"Wyklad {time.strftime('%Y-%m-%d %H:%M')}"
        return TranscriptSession(
            session_id=str(uuid.uuid4()),
            title=title,
            started_at=time.time(),
            speaker_map={0: SpeakerRole.LECTURER},
            metadata=metadata or {},
        )
