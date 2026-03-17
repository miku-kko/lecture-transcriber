from typing import Optional

from backend.models.transcript import TranscriptSegment


class TranscriptAssembler:
    """
    Maintains running transcript state.
    Merges interim results, replaces them with finals,
    and maintains consistent segment ordering.
    """

    def __init__(self):
        self._final_segments: list[TranscriptSegment] = []
        self._interim_segment: Optional[TranscriptSegment] = None

    def process_segment(self, segment: TranscriptSegment) -> dict:
        """
        Returns a dict with:
          - action: "update_interim" | "finalize"
          - segment: the segment to display
          - index: position in the segment list
        """
        if not segment.is_final:
            self._interim_segment = segment
            return {
                "action": "update_interim",
                "segment": segment,
                "index": len(self._final_segments),
            }
        else:
            self._final_segments.append(segment)
            self._interim_segment = None
            return {
                "action": "finalize",
                "segment": segment,
                "index": len(self._final_segments) - 1,
            }

    def get_full_transcript(self) -> list[TranscriptSegment]:
        return list(self._final_segments)

    def get_text_for_correction(self, window_size: int = 3) -> str:
        """Get recent finalized segments as context for correction."""
        recent = self._final_segments[-window_size:]
        return " ".join(s.text for s in recent)

    def reset(self) -> None:
        self._final_segments.clear()
        self._interim_segment = None
