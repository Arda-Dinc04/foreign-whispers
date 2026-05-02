"""AlignmentService: wraps VAD, diarization, and evaluation for the FastAPI layer."""
import logging
import subprocess

import soundfile as sf

from foreign_whispers.diarization import diarize_audio
from foreign_whispers.evaluation import clip_evaluation_report
from foreign_whispers.vad import detect_speech_activity as _detect

logger = logging.getLogger(__name__)


class AlignmentService:
    """Service providing VAD, diarization, and clip evaluation.

    Receives Settings via constructor so no global imports are needed.
    All heavy deps are optional — methods fall back to empty results gracefully.
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    def detect_speech_activity(self, audio_path: str) -> list[dict]:
        """Return [{start_s, end_s, label}]. Empty list if silero-vad absent."""
        return _detect(audio_path)

    def diarize(self, audio_path: str) -> list[dict]:
        """Return [{start_s, end_s, speaker}]. Empty list if pyannote absent or no token."""
        chunk_s = max(0, int(getattr(self._settings, "diarization_chunk_s", 120)))
        if not chunk_s:
            return diarize_audio(audio_path, hf_token=self._settings.hf_token or None)

        try:
            info = sf.info(audio_path)
            duration_s = float(info.frames) / float(info.samplerate)
        except Exception as exc:
            logger.warning("Could not inspect diarization audio %s: %s", audio_path, exc)
            return diarize_audio(audio_path, hf_token=self._settings.hf_token or None)

        if duration_s <= chunk_s:
            return diarize_audio(audio_path, hf_token=self._settings.hf_token or None)

        return self._diarize_in_chunks(audio_path, duration_s, chunk_s)

    def _diarize_in_chunks(self, audio_path: str, duration_s: float, chunk_s: int) -> list[dict]:
        """Diarize long audio in bounded chunks to avoid pyannote full-file hangs."""
        import tempfile
        from pathlib import Path

        all_segments: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="fw-diarize-") as tmp:
            tmp_dir = Path(tmp)
            for chunk_start in range(0, int(duration_s) + 1, chunk_s):
                if chunk_start >= duration_s:
                    break
                chunk_path = tmp_dir / f"chunk_{chunk_start}.wav"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(chunk_start),
                    "-t",
                    str(chunk_s),
                    "-i",
                    audio_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(chunk_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning("Skipping diarization chunk %ss: %s", chunk_start, proc.stderr)
                    continue

                chunk_segments = diarize_audio(str(chunk_path), hf_token=self._settings.hf_token or None)
                logger.info(
                    "Diarized chunk %ss of %s: %d segments",
                    chunk_start,
                    audio_path,
                    len(chunk_segments),
                )
                for segment in chunk_segments:
                    adjusted = dict(segment)
                    adjusted["start_s"] = float(adjusted["start_s"]) + chunk_start
                    adjusted["end_s"] = float(adjusted["end_s"]) + chunk_start
                    all_segments.append(adjusted)

        logger.info("Chunked diarization complete for %s: %d segments", audio_path, len(all_segments))
        return all_segments

    def evaluate_clip(self, metrics: list, aligned: list) -> dict:
        """Return a clip evaluation report dict."""
        return clip_evaluation_report(metrics, aligned)
