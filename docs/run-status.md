# Foreign Whispers Run Status

Last verified: 2026-05-01

## Local URLs

- API: http://localhost:8080
- Current development frontend: http://localhost:8504
- Docker frontend: http://localhost:8503

The public project docs mention `http://localhost:8501`, but this local compose
setup currently maps the frontend container to host port `8503`. Use `8504` for
the live source tree started with `PORT=8504 pnpm dev`.

## Workflow Readiness

| Stage | Status | Verification |
| --- | --- | --- |
| Download | Ready | `POST /api/download` returns cached video and captions |
| Transcribe | Ready | `POST /api/transcribe/GYQ5yGV_-Oc` returns transcript segments |
| Diarize | Code ready, optional runtime data | `POST /api/diarize/GYQ5yGV_-Oc` returns empty speakers without `FW_HF_TOKEN` |
| Translate | Ready | `POST /api/translate/GYQ5yGV_-Oc?target_language=es` returns Spanish segments |
| TTS | Ready for cached sample | `POST /api/tts/GYQ5yGV_-Oc?config=c-fb1074a&alignment=true` returns WAV path |
| Stitch | Ready | `POST /api/stitch/GYQ5yGV_-Oc?config=c-fb1074a` returns MP4 path |
| Captions | Ready | `GET /api/captions/GYQ5yGV_-Oc` returns WebVTT |
| Video playback | Ready | `GET /api/video/GYQ5yGV_-Oc?config=c-fb1074a` streams MP4 |

## Verified Artifacts

Artifacts are under `pipeline_data/api/`:

- `videos/` source MP4
- `youtube_captions/` caption text and JSON3
- `transcriptions/whisper/` English transcript JSON
- `translations/argos/` Spanish transcript JSON
- `diarizations/` optional diarization JSON/WAV; empty speakers without HF token
- `tts_audio/chatterbox/<config>/` dubbed WAV and `.align.json`
- `dubbed_videos/<config>/` final MP4
- `dubbed_captions/` final VTT captions

## Verification Commands

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/api/videos

VID=GYQ5yGV_-Oc
CFG=c-fb1074a
curl -X POST "http://localhost:8080/api/transcribe/$VID"
curl -X POST "http://localhost:8080/api/diarize/$VID"
curl -X POST "http://localhost:8080/api/translate/$VID?target_language=es"
curl -X POST "http://localhost:8080/api/tts/$VID?config=$CFG&alignment=true"
curl -X POST "http://localhost:8080/api/stitch/$VID?config=$CFG"
curl "http://localhost:8080/api/captions/$VID"
```

## Known Environment Boundary

Fresh uncached Whisper and Chatterbox GPU runs are blocked on this Mac/ARM host
because the documented GPU images do not publish `linux/arm64/v8` manifests.
Use an x86_64 Linux/NVIDIA host for a fully fresh GPU run, or keep using the
cached sample artifacts on this machine.

## Review Map

- Pipeline frontend flow: `frontend/src/hooks/use-pipeline.ts`
- Frontend stage display: `frontend/src/components/pipeline-table.tsx`,
  `frontend/src/components/pipeline-status-bar.tsx`
- API route registration: `api/src/main.py`
- Diarization API and merge: `api/src/routers/diarize.py`,
  `foreign_whispers/diarization.py`
- Translation shortening: `foreign_whispers/reranking.py`
- Alignment optimizer and metrics: `foreign_whispers/alignment.py`,
  `foreign_whispers/evaluation.py`
- TTS voice selection: `api/src/routers/tts.py`,
  `api/src/services/tts_service.py`, `api/src/services/tts_engine.py`,
  `foreign_whispers/voice_resolution.py`
- Stitch/caption output: `api/src/routers/stitch.py`
- Executed notebooks: `notebooks/*/*_executed.ipynb`
