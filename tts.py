"""Compatibility wrapper for the TTS engine.

The implementation lives in ``api.src.services.tts_engine``.  Some notebooks
and older tests still import ``tts`` from the repository root.
"""

from api.src.services.tts_engine import *  # noqa: F401,F403
