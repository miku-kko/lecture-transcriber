from datetime import datetime

from backend.models.transcript import (
    SpeakerRole,
    TranscriptSegment,
    TranscriptSession,
)

SPEAKER_EMOJI = {
    SpeakerRole.LECTURER: "🎓",
    SpeakerRole.STUDENT: "🙋",
    SpeakerRole.UNKNOWN: "❓",
}

SPEAKER_LABEL = {
    SpeakerRole.LECTURER: "Wykladowca",
    SpeakerRole.STUDENT: "Student",
    SpeakerRole.UNKNOWN: "Nieznany",
}

MONTH_NAMES_PL = {
    1: "01-Styczen",
    2: "02-Luty",
    3: "03-Marzec",
    4: "04-Kwiecien",
    5: "05-Maj",
    6: "06-Czerwiec",
    7: "07-Lipiec",
    8: "08-Sierpien",
    9: "09-Wrzesien",
    10: "10-Pazdziernik",
    11: "11-Listopad",
    12: "12-Grudzien",
}


class MarkdownWriter:
    def generate(self, session: TranscriptSession) -> str:
        parts = []
        parts.append(self._front_matter(session))
        parts.append(f"# {session.title}\n")
        parts.append(self._metadata_block(session))
        parts.append("## Transkrypcja\n")
        parts.append(self._transcript_body(session))

        corrected = [s for s in session.segments if s.correction and s.correction.items]
        if corrected:
            parts.append("\n## Korekty\n")
            parts.append(self._corrections_summary(corrected))

        return "\n".join(parts)

    def _front_matter(self, session: TranscriptSession) -> str:
        dt = datetime.fromtimestamp(session.started_at)
        duration = ""
        if session.ended_at:
            mins = int((session.ended_at - session.started_at) / 60)
            duration = f"{mins} min"

        speakers = sorted(
            set(SPEAKER_LABEL[s.speaker_role] for s in session.segments)
        )

        safe_title = session.title.replace('\\', '\\\\').replace('"', '\\"')

        extra = ""
        for k, v in session.metadata.items():
            safe_v = str(v).replace('\\', '\\\\').replace('"', '\\"')
            extra += f'{k}: "{safe_v}"\n'

        return f"""---
title: "{safe_title}"
date: {dt.strftime('%Y-%m-%d')}
time: {dt.strftime('%H:%M')}
duration: "{duration}"
type: lecture
speakers: [{', '.join(speakers)}]
tags: [wyklad, transkrypcja]
{extra}---
"""

    def _metadata_block(self, session: TranscriptSession) -> str:
        dt = datetime.fromtimestamp(session.started_at)
        return f"""| | |
|---|---|
| **Data** | {dt.strftime('%d.%m.%Y')} |
| **Godzina** | {dt.strftime('%H:%M')} |
| **Mowcy** | {len(set(s.speaker for s in session.segments))} |

"""

    def _transcript_body(self, session: TranscriptSession) -> str:
        lines = []
        current_speaker = None
        # Track which batch corrections we've already emitted
        emitted_batch_ids: set[str] = set()

        for seg in session.segments:
            if seg.correction and seg.is_corrected:
                batch_id = seg.correction.segment_id
                # For batch corrections (pipe-separated IDs), emit the combined
                # corrected text only once for the first segment in the batch
                if "|" in batch_id:
                    if batch_id in emitted_batch_ids:
                        continue
                    emitted_batch_ids.add(batch_id)

            if seg.speaker_role != current_speaker:
                current_speaker = seg.speaker_role
                emoji = SPEAKER_EMOJI.get(current_speaker, "")
                label = SPEAKER_LABEL.get(current_speaker, "Mowca")
                lines.append(f"\n**{emoji} {label}:**\n")

            text = seg.text
            if seg.correction and seg.is_corrected:
                text = seg.correction.corrected_text

            lines.append(f"> {text}\n")

        return "\n".join(lines)

    def _corrections_summary(self, segments: list[TranscriptSegment]) -> str:
        lines = []
        for seg in segments:
            if not seg.correction:
                continue
            for item in seg.correction.items:
                severity_icon = {
                    "error": "🔴",
                    "warning": "🟡",
                    "info": "🔵",
                }.get(item.severity.value, "")

                lines.append(
                    f"- {severity_icon} ~~{item.original_text}~~ "
                    f"-> **{item.suggested_text}** "
                    f"-- _{item.explanation}_"
                )
        return "\n".join(lines)
