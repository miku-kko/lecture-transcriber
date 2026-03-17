import logging
import os
import re
from datetime import datetime
from pathlib import Path

from backend.models.transcript import TranscriptSession
from backend.storage.markdown_writer import MONTH_NAMES_PL, MarkdownWriter

logger = logging.getLogger(__name__)


class VaultManager:
    def __init__(self, vault_path: str):
        self._vault_path = Path(vault_path)
        self._writer = MarkdownWriter()

    def initialize_vault(self) -> None:
        (self._vault_path / "Lectures").mkdir(parents=True, exist_ok=True)
        (self._vault_path / "Templates").mkdir(parents=True, exist_ok=True)

        index_path = self._vault_path / "_index.md"
        if not index_path.exists():
            index_path.write_text(
                "# Notatki z Wykladow\n\n"
                "Ten vault zawiera automatyczne transkrypcje wykladow.\n",
                encoding="utf-8",
            )
        logger.info(f"Vault initialized at {self._vault_path}")

    def save_session(self, session: TranscriptSession) -> Path:
        dt = datetime.fromtimestamp(session.started_at)

        year_dir = self._vault_path / "Lectures" / str(dt.year)
        month_dir = year_dir / MONTH_NAMES_PL[dt.month]
        month_dir.mkdir(parents=True, exist_ok=True)

        safe_title = self._sanitize_filename(session.title)
        filename = f"{dt.strftime('%Y-%m-%d')}_{safe_title}.md"
        filepath = month_dir / filename

        counter = 1
        while filepath.exists():
            filename = f"{dt.strftime('%Y-%m-%d')}_{safe_title}_{counter}.md"
            filepath = month_dir / filename
            counter += 1

        content = self._writer.generate(session)
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Session saved to {filepath}")
        return filepath

    def list_lectures(self) -> list[dict]:
        lectures = []
        lecture_dir = self._vault_path / "Lectures"
        if not lecture_dir.exists():
            return lectures

        for md_file in sorted(lecture_dir.rglob("*.md"), reverse=True):
            stat = md_file.stat()
            lectures.append(
                {
                    "path": str(md_file.relative_to(self._vault_path)),
                    "filename": md_file.name,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        return lectures

    def read_lecture(self, relative_path: str) -> str:
        filepath = (self._vault_path / relative_path).resolve()
        vault_root = str(self._vault_path.resolve()) + os.sep
        if not str(filepath).startswith(vault_root):
            raise ValueError("Access denied: path traversal detected")
        return filepath.read_text(encoding="utf-8")

    @staticmethod
    def _sanitize_filename(title: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*]', "", title)
        safe = safe.replace(" ", "-")
        return safe[:80]
