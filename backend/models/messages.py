import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class WSMessageType(Enum):
    # Server -> Client
    TRANSCRIPT_INTERIM = "transcript_interim"
    TRANSCRIPT_FINAL = "transcript_final"
    CORRECTION_RESULT = "correction_result"
    SESSION_STARTED = "session_started"
    SESSION_STOPPED = "session_stopped"
    RAG_RESULT = "rag_result"
    ERROR = "error"
    STATUS = "status"
    LECTURE_LIST = "lecture_list"

    # Client -> Server
    CMD_START_RECORDING = "cmd_start_recording"
    CMD_STOP_RECORDING = "cmd_stop_recording"
    CMD_SET_MODE = "cmd_set_mode"
    CMD_ACCEPT_CORRECTION = "cmd_accept_correction"
    CMD_REJECT_CORRECTION = "cmd_reject_correction"
    CMD_EDIT_SEGMENT = "cmd_edit_segment"
    CMD_RAG_QUERY = "cmd_rag_query"
    CMD_LIST_LECTURES = "cmd_list_lectures"
    CMD_SET_SPEAKER_ROLE = "cmd_set_speaker_role"
    CMD_SET_TITLE = "cmd_set_title"
    CMD_SET_KEYTERMS = "cmd_set_keyterms"
    CMD_SET_GAIN = "cmd_set_gain"

    # Server -> Client (audio level)
    AUDIO_LEVEL = "audio_level"


@dataclass
class WSMessage:
    type: WSMessageType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    message_id: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        })

    @staticmethod
    def from_json(raw: str) -> "WSMessage":
        data = json.loads(raw)
        return WSMessage(
            type=WSMessageType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id"),
        )
