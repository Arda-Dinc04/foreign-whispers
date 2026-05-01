import pytest

from foreign_whispers.diarization import assign_speakers, diarize_audio


def test_returns_empty_without_token():
    result = diarize_audio("/any/path.wav", hf_token=None)
    assert result == []


def test_returns_empty_with_empty_token():
    result = diarize_audio("/any/path.wav", hf_token="")
    assert result == []


def test_assign_speakers_uses_max_overlap_and_does_not_mutate():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
    ]
    diarization = [
        {"start_s": 0.0, "end_s": 1.0, "speaker": "SPEAKER_00"},
        {"start_s": 1.0, "end_s": 4.0, "speaker": "SPEAKER_01"},
    ]

    result = assign_speakers(segments, diarization)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"
    assert "speaker" not in segments[0]


def test_assign_speakers_defaults_to_speaker_00_when_no_overlap():
    result = assign_speakers(
        [{"start": 10.0, "end": 11.0, "text": "late"}],
        [{"start_s": 0.0, "end_s": 1.0, "speaker": "SPEAKER_00"}],
    )
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_rejects_invalid_segment():
    with pytest.raises(ValueError):
        assign_speakers([{"start": 2.0, "end": 2.0, "text": "bad"}], [])


def test_returns_empty_when_pyannote_absent(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    result = diarize_audio("/any/path.wav", hf_token="fake-token")
    assert result == []


@pytest.mark.requires_pyannote
def test_real_diarization_returns_speaker_labels(tmp_path):
    """Integration test — requires pyannote.audio and FW_HF_TOKEN env var."""
    import os
    token = os.environ.get("FW_HF_TOKEN")
    if not token:
        pytest.skip("FW_HF_TOKEN not set")
    result = diarize_audio("/path/to/sample.wav", hf_token=token)
    assert isinstance(result, list)
    for r in result:
        assert "start_s" in r and "end_s" in r and "speaker" in r
