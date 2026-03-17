"""Basic tests for data models."""
import time
from backend.models.transcript import TranscriptSegment, TranscriptSession, SpeakerRole, Word
from backend.models.correction import CorrectionResult, CorrectionItem, CorrectionType, CorrectionSeverity
from backend.models.messages import WSMessage, WSMessageType


def test_transcript_segment_to_dict():
    seg = TranscriptSegment(
        segment_id="test-1",
        speaker=0,
        speaker_role=SpeakerRole.LECTURER,
        text="Witajcie na wykladzie",
        words=[],
        timestamp=time.time(),
        is_final=True,
    )
    d = seg.to_dict()
    assert d["segment_id"] == "test-1"
    assert d["speaker_role"] == "lecturer"
    assert d["is_final"] is True


def test_transcript_session_create():
    session = TranscriptSession.create("Test wyklad")
    assert session.title == "Test wyklad"
    assert session.session_id
    assert session.started_at > 0
    assert session.speaker_map[0] == SpeakerRole.LECTURER


def test_transcript_session_default_title():
    session = TranscriptSession.create()
    assert "Wyklad" in session.title


def test_correction_result_to_dict():
    item = CorrectionItem(
        original_text="bledny",
        suggested_text="prawidlowy",
        correction_type=CorrectionType.SPELLING,
        severity=CorrectionSeverity.WARNING,
        explanation="Blad ortograficzny",
    )
    result = CorrectionResult(
        segment_id="seg-1",
        corrected_text="prawidlowy tekst",
        items=[item],
        model_used="test-model",
        processing_time_ms=100.0,
        confidence=0.9,
    )
    d = result.to_dict()
    assert d["segment_id"] == "seg-1"
    assert len(d["items"]) == 1
    assert d["items"][0]["type"] == "spelling"


def test_ws_message_roundtrip():
    msg = WSMessage(
        type=WSMessageType.TRANSCRIPT_FINAL,
        payload={"text": "test"},
    )
    json_str = msg.to_json()
    parsed = WSMessage.from_json(json_str)
    assert parsed.type == WSMessageType.TRANSCRIPT_FINAL
    assert parsed.payload["text"] == "test"


def test_speaker_role_values():
    assert SpeakerRole.LECTURER.value == "lecturer"
    assert SpeakerRole.STUDENT.value == "student"
    assert SpeakerRole.UNKNOWN.value == "unknown"


def test_vault_path_traversal():
    """Test that vault_manager blocks path traversal."""
    from backend.storage.vault_manager import VaultManager
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VaultManager(tmpdir)
        try:
            vm.read_lecture("../../etc/passwd")
            assert False, "Should have raised an error"
        except (ValueError, FileNotFoundError):
            pass  # Expected
