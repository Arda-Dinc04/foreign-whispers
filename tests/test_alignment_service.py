from unittest.mock import MagicMock
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.alignment import compute_segment_metrics, global_align


def _svc(hf_token=""):
    settings = MagicMock()
    settings.hf_token = hf_token
    settings.diarization_chunk_s = 120
    return AlignmentService(settings)


def test_detect_returns_list():
    result = _svc().detect_speech_activity("/nonexistent.wav")
    assert isinstance(result, list)


def test_diarize_without_token_returns_empty():
    result = _svc(hf_token="").diarize("/nonexistent.wav")
    assert result == []


def test_diarize_long_audio_uses_chunks(monkeypatch):
    calls = []

    class FakeInfo:
        frames = 16000 * 250
        samplerate = 16000

    monkeypatch.setattr("api.src.services.alignment_service.sf.info", lambda path: FakeInfo())

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("api.src.services.alignment_service.subprocess.run", fake_run)

    def fake_diarize(path, hf_token=None):
        start = 0.0 if "chunk_0" in path else 1.0
        return [{"start_s": start, "end_s": start + 2.0, "speaker": "SPEAKER_00"}]

    monkeypatch.setattr("api.src.services.alignment_service.diarize_audio", fake_diarize)

    result = _svc(hf_token="token").diarize("/tmp/audio.wav")

    assert len(calls) == 3
    assert result[0]["start_s"] == 0.0
    assert result[1]["start_s"] == 121.0
    assert result[2]["start_s"] == 241.0


def test_evaluate_clip_returns_dict():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hello"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hola"}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = _svc().evaluate_clip(metrics, aligned)
    assert "mean_abs_duration_error_s" in report
