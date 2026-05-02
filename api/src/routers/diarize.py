"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import functools
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        _merge_speakers_into_transcription(title, data.get("segments", []))
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found for {video_id}")

    audio_path = diar_dir / f"{title}.wav"
    if not audio_path.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            str(audio_path),
        ]
        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            functools.partial(subprocess.run, cmd, capture_output=True, text=True),
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg audio extraction failed: {proc.stderr}")

    loop = asyncio.get_event_loop()
    diar_segments = await loop.run_in_executor(None, _alignment_service.diarize, str(audio_path))
    speakers = sorted({s["speaker"] for s in diar_segments if s.get("speaker")})

    _merge_speakers_into_transcription(title, diar_segments)

    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result, indent=2))

    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)


def _merge_speakers_into_transcription(title: str, diar_segments: list[dict]) -> None:
    if not diar_segments:
        return

    trans_path = settings.transcriptions_dir / f"{title}.json"
    if not trans_path.exists():
        return

    transcript = json.loads(trans_path.read_text())
    transcript["segments"] = assign_speakers(transcript.get("segments", []), diar_segments)
    trans_path.write_text(json.dumps(transcript, indent=2))
