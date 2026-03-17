from __future__ import annotations

import logging
import time
from typing import Optional

from backend.models.transcript import SpeakerRole, TranscriptSession

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self._current_session: Optional[TranscriptSession] = None
        self._mode: str = "automatic"

    def create_session(
        self, title: str = "", metadata: dict | None = None
    ) -> TranscriptSession:
        self._current_session = TranscriptSession.create(title, metadata)
        logger.info(
            f"Session created: {self._current_session.session_id} "
            f"'{self._current_session.title}'"
        )
        return self._current_session

    def end_session(self) -> Optional[TranscriptSession]:
        if self._current_session:
            self._current_session.ended_at = time.time()
            logger.info(f"Session ended: {self._current_session.session_id}")
        session = self._current_session
        self._current_session = None
        return session

    @property
    def current(self) -> Optional[TranscriptSession]:
        return self._current_session

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("automatic", "interactive"):
            raise ValueError(f"Invalid mode: {value}")
        self._mode = value
        logger.info(f"Mode set to: {value}")

    @property
    def is_active(self) -> bool:
        return self._current_session is not None

    def set_speaker_role(self, speaker_id: int, role: str) -> None:
        if self._current_session:
            self._current_session.speaker_map[speaker_id] = SpeakerRole(role)

    def set_title(self, title: str) -> None:
        if self._current_session:
            self._current_session.title = title
