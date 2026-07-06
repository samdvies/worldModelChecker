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
