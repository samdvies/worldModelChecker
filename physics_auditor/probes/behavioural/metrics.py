"""Threshold-free metrics for the behavioural VoE probe (spec §5.1)."""
from collections.abc import Sequence


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """AUROC of scores separating pos (violate) from neg (obey).

    Mann-Whitney formulation: fraction of (pos, neg) pairs where pos > neg,
    ties counting 0.5. Avoids an sklearn dependency.
    """
    if not pos or not neg:
        raise ValueError("auroc requires at least one score in each class")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def distinct_score_count(obey: Sequence[float], violate: Sequence[float]) -> int:
    """Number of distinct score values across BOTH the obey and violate
    score lists combined.

    An AUROC computed over n eval pairs is only as informative as its
    effective sample size: if the scoring function collapses to a handful
    of distinct values across many "different" seeds (e.g. a pixel-MSE
    baseline that is translation-invariant along the only dimension the
    generator randomises), the true effective n is much smaller than
    len(obey) + len(violate) even though the reported AUROC uses all of
    them. Report this alongside AUROC so degenerate cells are visible.
    """
    return len(set(obey) | set(violate))
