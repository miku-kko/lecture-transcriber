from dataclasses import dataclass, field
from enum import Enum


class CorrectionType(Enum):
    GRAMMAR = "grammar"
    SPELLING = "spelling"
    LOGIC = "logic"
    PUNCTUATION = "punctuation"
    STYLE = "style"


class CorrectionSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CorrectionItem:
    original_text: str
    suggested_text: str
    correction_type: CorrectionType
    severity: CorrectionSeverity
    explanation: str
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict:
        return {
            "original": self.original_text,
            "suggested": self.suggested_text,
            "type": self.correction_type.value,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class CorrectionResult:
    segment_id: str
    corrected_text: str
    items: list[CorrectionItem] = field(default_factory=list)
    model_used: str = ""
    processing_time_ms: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "corrected_text": self.corrected_text,
            "items": [i.to_dict() for i in self.items],
            "model": self.model_used,
            "processing_ms": self.processing_time_ms,
            "confidence": self.confidence,
        }
