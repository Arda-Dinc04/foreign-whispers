"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import math
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    budget = max(1, math.floor(target_duration_s * 15))
    baseline = _clean_spacing(baseline_es)
    if not baseline:
        return []

    candidates: dict[str, str] = {}

    def add(text: str, rationale: str) -> None:
        cleaned = _clean_spacing(text)
        if cleaned and len(cleaned) < len(baseline):
            candidates.setdefault(cleaned, rationale)

    add(_apply_phrase_shortening(baseline), "contracted common Spanish phrases")
    no_fillers = _remove_fillers(baseline)
    add(no_fillers, "removed filler words")
    add(_apply_phrase_shortening(no_fillers), "removed filler words and contracted phrases")
    add(_remove_redundant_adverbs(no_fillers), "removed redundant adverbs")
    add(_remove_parentheticals(no_fillers), "removed parenthetical aside")

    # Last-resort concise candidate: trim trailing subordinate clauses.  This is
    # intentionally conservative and only cuts at punctuation/conjunctions.
    for splitter in [", que ", ", lo que ", " porque ", " ya que ", " mientras "]:
        if splitter in no_fillers.lower():
            idx = no_fillers.lower().find(splitter)
            add(no_fillers[:idx], f"trimmed trailing clause after {splitter.strip()!r}")

    ranked = [
        TranslationCandidate(text=text, char_count=len(text), brevity_rationale=rationale)
        for text, rationale in candidates.items()
    ]
    ranked.sort(key=lambda c: (len(c.text) > budget, abs(len(c.text) - budget), len(c.text), c.text.lower()))
    return ranked


_PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("en este momento", "ahora"),
    ("en estos momentos", "ahora"),
    ("en el momento actual", "ahora"),
    ("a continuación", "luego"),
    ("con el fin de", "para"),
    ("con la finalidad de", "para"),
    ("debido a que", "porque"),
    ("a causa de que", "porque"),
    ("por lo tanto", "así que"),
    ("sin embargo", "pero"),
    ("de alguna manera", ""),
    ("por supuesto", "claro"),
    ("en realidad", ""),
    ("básicamente", ""),
    ("realmente", ""),
    ("actualmente", "hoy"),
    ("aproximadamente", "casi"),
    ("alrededor de", "casi"),
    ("un montón de", "muchos"),
    ("una gran cantidad de", "muchos"),
    ("cada uno de", "cada"),
)

_FILLER_WORDS = {
    "bueno",
    "pues",
    "entonces",
    "realmente",
    "básicamente",
    "simplemente",
    "probablemente",
    "posiblemente",
    "claramente",
}


def _clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([¿¡])\s+", r"\1", text)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", text)
    return text


def _replace_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)


def _apply_phrase_shortening(text: str) -> str:
    result = text
    for old, new in _PHRASE_REPLACEMENTS:
        result = _replace_case_insensitive(result, old, new)
    return _clean_spacing(result)


def _remove_fillers(text: str) -> str:
    pattern = r"\b(" + "|".join(re.escape(w) for w in sorted(_FILLER_WORDS)) + r")\b"
    return _clean_spacing(re.sub(pattern, "", text, flags=re.IGNORECASE))


def _remove_redundant_adverbs(text: str) -> str:
    return _clean_spacing(re.sub(r"\b\w+mente\b", "", text, flags=re.IGNORECASE))


def _remove_parentheticals(text: str) -> str:
    return _clean_spacing(re.sub(r"\([^)]*\)", "", text))
