"""Threshold-free metrics for the behavioural VoE probe (spec §5.1)."""
from collections.abc import Sequence

import numpy as np


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


def bootstrap_auroc_ci(
    obey_scores: Sequence[float],
    violate_scores: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap 100*(1-alpha)% CI for the VoE AUROC.

    obey_scores[i] and violate_scores[i] come from the same clip pair i
    (spec §5.1's paired MinimalPairs) -- each bootstrap resample draws pair
    *indices* with replacement (not obey/violate scores independently), so
    the obey/violate pairing within a resampled index is always preserved.

    Degenerate resamples (e.g. all resampled scores identical) fall out of
    auroc()'s own tie handling -- a wholly-tied comparison scores 0.5, same
    convention as the point estimate -- so no special-casing is needed here.
    """
    if len(obey_scores) != len(violate_scores):
        raise ValueError("obey_scores and violate_scores must be paired (equal length)")
    n = len(obey_scores)
    if n == 0:
        raise ValueError("bootstrap_auroc_ci requires at least one pair")

    obey = np.asarray(obey_scores, dtype=float)
    violate = np.asarray(violate_scores, dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    boot_aurocs = [auroc(pos=list(violate[row]), neg=list(obey[row])) for row in idx]

    lo, hi = np.percentile(boot_aurocs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
