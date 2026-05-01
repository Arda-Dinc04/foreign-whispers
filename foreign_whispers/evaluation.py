"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
            "n_overlaps":                0,
            "n_severe_stretch":          0,
            "action_counts":             {},
            "semantic_similarity":       None,
            "naturalness_score":         None,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )
    n_overlaps = sum(
        1
        for prev, cur in zip(aligned, aligned[1:])
        if cur.scheduled_start < prev.scheduled_end
    )
    action_counts = {
        action.value: sum(1 for a in aligned if a.action == action)
        for action in AlignAction
    }
    speed_factors = [a.stretch_factor for a in aligned if a.stretch_factor > 0]
    naturalness = None
    if len(speed_factors) >= 2:
        naturalness = round(max(0.0, 1.0 - _stats.pstdev(speed_factors)), 3)
    elif speed_factors:
        naturalness = 1.0

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
        "n_overlaps":                n_overlaps,
        "n_severe_stretch":          n_severe,
        "action_counts":             action_counts,
        "semantic_similarity":       None,
        "naturalness_score":         naturalness,
    }
