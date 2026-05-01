"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging
import functools
import os
from collections import namedtuple

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        _patch_torchaudio_metadata_type()
        from pyannote.audio import Pipeline
    except (ImportError, TypeError, AttributeError) as exc:
        logger.warning("pyannote.audio unavailable (%s) — returning empty diarization.", exc)
        return []

    try:
        _patch_torch_load_for_pyannote()
        import soundfile as sf
        import torch

        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono_samples = samples.mean(axis=1)
        waveform = torch.from_numpy(mono_samples).unsqueeze(0)
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def _patch_torch_load_for_pyannote() -> None:
    """Allow trusted pyannote checkpoints to load under PyTorch 2.6+."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        import torch
        from torch.torch_version import TorchVersion
    except ImportError:
        return

    try:
        torch.serialization.add_safe_globals([TorchVersion])
    except (AttributeError, TypeError):
        pass

    if getattr(torch.load, "_foreign_whispers_weights_patch", False):
        return

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    patched_load._foreign_whispers_weights_patch = True  # type: ignore[attr-defined]
    torch.load = patched_load


def _patch_torchaudio_metadata_type() -> None:
    """Patch torchaudio 2.10 compatibility for pyannote.audio 3.x imports.

    pyannote only uses ``torchaudio.AudioMetaData`` as a type annotation, but
    newer torchaudio builds no longer expose that symbol at the top level.
    """
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = namedtuple(  # type: ignore[attr-defined]
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


def _interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return positive overlap in seconds for two half-open intervals."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
    unknown_label: str = "SPEAKER_00",
) -> list[dict]:
    """Assign speaker labels to transcript segments by maximum temporal overlap.

    The input transcript uses ``start``/``end`` keys while pyannote output uses
    ``start_s``/``end_s``.  Returned segment dicts are copies so callers can
    preserve the original transcript when needed.
    """
    labeled: list[dict] = []

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            raise ValueError(f"Invalid segment timing: end ({end}) must be greater than start ({start})")

        best_speaker = unknown_label
        best_overlap = 0.0

        for diar in diarization:
            overlap = _interval_overlap(
                start,
                end,
                float(diar["start_s"]),
                float(diar["end_s"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(diar.get("speaker", unknown_label))

        labeled.append({**seg, "speaker": best_speaker})

    return labeled
