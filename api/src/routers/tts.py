"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(None, description="Reference voice WAV path, e.g. es/default.wav"),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")
    if not pathlib.Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Translated transcript not found")

    translated = _repair_translation_speakers(title, pathlib.Path(source_path))
    resolved_speaker_wav, voice_map = _resolve_voice_selection(translated, speaker_wav)

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        speaker_wav=resolved_speaker_wav,
        voice_map=voice_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")


def _resolve_voice_selection(
    translated: dict,
    requested_speaker_wav: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    if requested_speaker_wav:
        return requested_speaker_wav, None

    segments = translated.get("segments", [])
    speakers = sorted({seg.get("speaker") for seg in segments if seg.get("speaker")})
    if speakers:
        voice_map: dict[str, str] = {}
        for speaker in speakers:
            try:
                voice_map[speaker] = resolve_speaker_wav(settings.speakers_dir, "es", speaker)
            except FileNotFoundError:
                continue
        if voice_map:
            return None, voice_map

    try:
        return resolve_speaker_wav(settings.speakers_dir, "es"), None
    except FileNotFoundError:
        return None, None


def _repair_translation_speakers(title: str, translation_path: pathlib.Path) -> dict:
    """Ensure translated segments keep speaker labels from the source transcript.

    Older cached translations may have been generated before diarization ran.
    TTS consumes the translated JSON, so repair it in place before voice-map
    resolution.
    """
    translated = json.loads(translation_path.read_text())
    segments = translated.get("segments", [])
    if not segments or any(seg.get("speaker") for seg in segments):
        return translated

    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return translated

    transcript = json.loads(transcript_path.read_text())
    changed = False
    for source_seg, translated_seg in zip(transcript.get("segments", []), segments):
        speaker = source_seg.get("speaker")
        if speaker and not translated_seg.get("speaker"):
            translated_seg["speaker"] = speaker
            changed = True

    if changed:
        translation_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False))

    return translated
