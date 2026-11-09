"""Robust scale estimation by agreement between independent evidence.

Candidates come from many texts, each with several hypotheses (units, span
interpretations). Rather than trusting any single reading, we look for the
scale value that the largest weight of *distinct* texts agrees on. A text
contributes at most one vote (its best hypothesis) to a cluster, so a text with
many hypotheses cannot out-vote independent corroboration.

Agreement is scored with a Gaussian kernel, so tight agreement beats loose
agreement. Because drawings follow one dimensioning convention per sheet
(centrelines, clear interior faces, or outside faces), each convention is also
scored as its own pool and a coherent pool earns a bonus.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from .models import ScaleEvidence

_MIXED = "mixed"
_CONVENTIONS = ("centerline", "interior", "exterior")


@dataclass(frozen=True)
class Consensus:
    """The winning cluster.

    Attributes:
        meters_per_pixel: Weighted median of the member estimates.
        members: Indices (into the candidate list) of each group's best member.
        support: Number of distinct groups (texts) in the cluster.
        total_weight: Sum of the members' weights.
        spread: Weighted mean relative deviation of members from the estimate.
        score: Kernel-weighted agreement used to rank clusters.
        convention: The shared dimensioning convention, or ``"mixed"``.
    """

    meters_per_pixel: float
    members: tuple[int, ...]
    support: int
    total_weight: float
    spread: float
    score: float = 0.0
    convention: str = _MIXED


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median; ties resolve to the lower value for determinism."""
    if not values:
        raise ValueError("weighted_median requires at least one value")
    pairs = sorted(zip(values, weights, strict=True))
    half = 0.5 * sum(w for _, w in pairs)
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return value
    return pairs[-1][0]


def find_consensus(candidates: Sequence[ScaleEvidence], tolerance: float) -> Consensus | None:
    """Best-supported cluster within ``tolerance`` (relative), or ``None``."""
    usable = [i for i, c in enumerate(candidates) if c.meters_per_pixel > 0 and c.weight > 0]
    if not usable:
        return None
    order = sorted(usable, key=lambda i: candidates[i].meters_per_pixel)
    scales = [candidates[i].meters_per_pixel for i in order]
    sigma = tolerance / 2.0

    best: tuple[float, int, float] | None = None
    best_members: dict[int, int] = {}
    for pivot in scales:
        low = bisect_left(scales, pivot / (1.0 + tolerance))
        high = bisect_right(scales, pivot * (1.0 + tolerance))
        per_group: dict[int, tuple[float, int]] = {}
        for idx in order[low:high]:
            candidate = candidates[idx]
            deviation = (candidate.meters_per_pixel - pivot) / pivot
            contribution = candidate.weight * math.exp(-0.5 * (deviation / sigma) ** 2)
            current = per_group.get(candidate.group)
            if current is None or contribution > current[0]:
                per_group[candidate.group] = (contribution, idx)
        score = sum(c for c, _ in per_group.values())
        key = (round(score, 9), len(per_group), -pivot)
        if best is None or key > best:
            best = key
            best_members = {group: idx for group, (_, idx) in per_group.items()}

    members = sorted(best_members.values())
    values = [candidates[i].meters_per_pixel for i in members]
    weights = [candidates[i].weight for i in members]
    estimate = weighted_median(values, weights)
    total = sum(weights)
    spread = sum(w * abs(v - estimate) / estimate for v, w in zip(values, weights, strict=True))
    assert best is not None
    return Consensus(
        meters_per_pixel=estimate,
        members=tuple(members),
        support=len(members),
        total_weight=total,
        spread=spread / total if total else 0.0,
        score=best[0],
        convention=_shared_convention([candidates[i] for i in members]),
    )


def find_best_consensus(
    candidates: Sequence[ScaleEvidence], tolerance: float, coherence_bonus: float
) -> Consensus | None:
    """Consensus over the whole pool and over each single-convention pool.

    A single-convention pool contains that convention's hypotheses plus
    convention-neutral evidence (drawing ratio). Its score is multiplied by
    ``coherence_bonus``; the highest-scoring pool wins.
    """
    results: list[Consensus] = []
    mixed = find_consensus(candidates, tolerance)
    if mixed is not None:
        results.append(mixed)
    for convention in _CONVENTIONS:
        indices = [i for i, c in enumerate(candidates) if c.convention in (convention, "")]
        if not indices or len(indices) == len(candidates):
            continue
        found = find_consensus([candidates[i] for i in indices], tolerance)
        if found is None:
            continue
        results.append(
            Consensus(
                meters_per_pixel=found.meters_per_pixel,
                members=tuple(indices[m] for m in found.members),
                support=found.support,
                total_weight=found.total_weight,
                spread=found.spread,
                score=found.score * coherence_bonus,
                convention=found.convention,
            )
        )
    if not results:
        return None
    return max(results, key=lambda r: (round(r.score, 9), r.support, -r.meters_per_pixel))


def _shared_convention(members: Sequence[ScaleEvidence]) -> str:
    conventions = {m.convention for m in members if m.convention}
    return next(iter(conventions)) if len(conventions) == 1 else _MIXED
