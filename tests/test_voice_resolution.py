import pytest

from foreign_whispers.voice_resolution import resolve_speaker_wav


def test_resolve_speaker_specific(tmp_path):
    speakers = tmp_path
    (speakers / "es").mkdir()
    (speakers / "es" / "SPEAKER_00.wav").write_bytes(b"RIFF")
    (speakers / "es" / "default.wav").write_bytes(b"RIFF")
    (speakers / "default.wav").write_bytes(b"RIFF")

    assert resolve_speaker_wav(speakers, "es", "SPEAKER_00") == "es/SPEAKER_00.wav"


def test_resolve_language_default_then_global(tmp_path):
    speakers = tmp_path
    (speakers / "es").mkdir()
    (speakers / "es" / "default.wav").write_bytes(b"RIFF")
    (speakers / "default.wav").write_bytes(b"RIFF")

    assert resolve_speaker_wav(speakers, "es", "SPEAKER_01") == "es/default.wav"
    assert resolve_speaker_wav(speakers, "fr", "SPEAKER_01") == "default.wav"


def test_resolve_raises_without_any_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_speaker_wav(tmp_path, "es", "SPEAKER_00")
