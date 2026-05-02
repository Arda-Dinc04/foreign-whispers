import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("whisper.load_model", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("TTS.api.TTS", lambda *a, **kw: MagicMock())

    from api.src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    from api.src.main import app

    with TestClient(app) as c:
        yield c


def test_diarize_extracts_caches_and_merges_transcript(client, monkeypatch, tmp_path):
    title = "Test Title"
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / f"{title}.mp4").write_bytes(b"video")
    trans_dir = tmp_path / "transcriptions" / "whisper"
    trans_dir.mkdir(parents=True)
    trans_path = trans_dir / f"{title}.json"
    trans_path.write_text(json.dumps({
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
        "text": "hello",
    }))

    monkeypatch.setattr("api.src.routers.diarize.resolve_title", lambda video_id: title)

    def fake_run(cmd, capture_output=True, text=True):
        audio_path = cmd[-1]
        open(audio_path, "wb").write(b"RIFF")
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("api.src.routers.diarize.subprocess.run", fake_run)
    monkeypatch.setattr(
        "api.src.routers.diarize._alignment_service.diarize",
        lambda audio_path: [{"start_s": 0.0, "end_s": 2.0, "speaker": "SPEAKER_00"}],
    )

    resp = client.post("/api/diarize/G3Eup4mfJdA")

    assert resp.status_code == 200
    body = resp.json()
    assert body["speakers"] == ["SPEAKER_00"]
    assert body["skipped"] is False
    assert (tmp_path / "diarizations" / f"{title}.json").exists()
    merged = json.loads(trans_path.read_text())
    assert merged["segments"][0]["speaker"] == "SPEAKER_00"


def test_diarize_returns_cached_result(client, monkeypatch, tmp_path):
    title = "Cached"
    diar_dir = tmp_path / "diarizations"
    diar_dir.mkdir()
    (diar_dir / f"{title}.json").write_text(json.dumps({
        "speakers": ["SPEAKER_00"],
        "segments": [{"start_s": 0.0, "end_s": 1.0, "speaker": "SPEAKER_00"}],
    }))
    trans_dir = tmp_path / "transcriptions" / "whisper"
    trans_dir.mkdir(parents=True)
    trans_path = trans_dir / f"{title}.json"
    trans_path.write_text(json.dumps({
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "text": "hello",
    }))
    monkeypatch.setattr("api.src.routers.diarize.resolve_title", lambda video_id: title)

    resp = client.post("/api/diarize/G3Eup4mfJdA")

    assert resp.status_code == 200
    assert resp.json()["skipped"] is True
    merged = json.loads(trans_path.read_text())
    assert merged["segments"][0]["speaker"] == "SPEAKER_00"
