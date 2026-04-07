"""
Scoring functions for content value and staleness.
"""

import math


def compute_staleness_score(age_days: float, half_life_days: float = 90.0) -> float:
    """Exponential staleness decay.  Returns 0.0 (fresh) → ~1.0 (stale)."""
    return 1.0 - math.exp(-age_days / max(half_life_days, 1e-9))


def compute_value_score(
    hit_count: int = 0,
    content_density: float = 0.5,
    age_days: float = 0.0,
    near_dup_sim: float = 0.0,
    source_priority: float = 1.0,
    half_life_days: float = 90.0,
    max_hit_count: float = 100.0,
) -> float:
    """Composite value score (0–1 * source_priority).  Higher = more valuable.

    v = 0.35*utility + 0.20*quality + 0.25*freshness + 0.20*uniqueness
    """
    utility = min(1.0, math.log1p(hit_count) / math.log1p(max(max_hit_count, 1.0)))
    quality = max(0.0, content_density * (1.0 - near_dup_sim))
    freshness = math.exp(-age_days / max(half_life_days, 1e-9))
    uniqueness = max(0.0, 1.0 - near_dup_sim)
    raw = 0.35 * utility + 0.20 * quality + 0.25 * freshness + 0.20 * uniqueness
    return max(0.0, min(1.0, raw * source_priority))
