"""Causal-intervention probes with a placebo control (spec 5.3): patch a
stack's context latents along the (violate-minus-obey) concept direction,
closed-loop roll out with the stack's own trained predictor, and check
whether the patched rollout moves TOWARD the true counterfactual (the
violate clip's observed latents) more than a random-direction placebo of
the same magnitude does. numpy only -- no sklearn/scipy; bootstrap CIs are
hand-rolled with a seeded rng."""
from dataclasses import dataclass, field

import numpy as np

from physics_auditor.probes.mechanistic.data import latent_frames, probe_pairs

HORIZON = 8
K_CONTEXT = 4
CONCEPT_FRAME_OFFSETS = range(1, 4)  # cf+1 .. cf+3 inclusive
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-dim L2 distance over frames: mean_t ||a_t - b_t||_2, for
    a, b of shape (T, D)."""
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def _direction_from_diffs(diffs: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure helper: given (N, D) per-frame (violate - obey) differences,
    returns (unit_direction, alpha) where alpha is the norm of the raw
    difference-of-means vector -- this is what's unit-tested against a
    synthetic linear system where the true direction is known."""
    raw = np.mean(diffs, axis=0)
    alpha = float(np.linalg.norm(raw))
    unit = raw / alpha if alpha > 1e-12 else raw
    return unit.astype(np.float64), alpha


def concept_direction(train_pairs, encoder, cache_dir: str = "cache") -> tuple[np.ndarray, float]:
    """Difference-of-means concept direction (violate - obey), computed at
    frames cf+1..cf+3 over probe-TRAIN pairs, normalised to unit; alpha is
    the norm of the raw (unnormalised) difference-of-means so the patch
    magnitude matches the real effect size."""
    diffs = []
    for pair in train_pairs:
        obey_z, violate_z = latent_frames(pair, encoder, cache_dir=cache_dir)
        cf = pair.critical_frame
        for off in CONCEPT_FRAME_OFFSETS:
            idx = cf + off
            if idx < 0 or idx >= obey_z.shape[0] or idx >= violate_z.shape[0]:
                continue
            diffs.append(violate_z[idx] - obey_z[idx])
    return _direction_from_diffs(np.asarray(diffs, dtype=np.float64))


def patched_rollout(predictor, context: np.ndarray, direction: np.ndarray, alpha: float, horizon: int = HORIZON) -> np.ndarray:
    """Patches ALL k context latents with +alpha*direction, then rolls out
    `horizon` steps in closed loop with the stack's predictor."""
    patched_ctx = (context.astype(np.float64) + alpha * direction).astype(np.float32)
    return predictor.rollout(patched_ctx, horizon)


def effect_score(rollout: np.ndarray, obey_observed: np.ndarray, violate_observed: np.ndarray) -> float:
    """Positive means the patched rollout moved TOWARD the true
    counterfactual (violate_observed) relative to the obey/violate gap."""
    d_obey = dist(rollout, obey_observed)
    d_violate = dist(rollout, violate_observed)
    d_gap = dist(obey_observed, violate_observed)
    return (d_obey - d_violate) / (d_gap + 1e-9)


def paired_bootstrap_ci(diffs: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = 0, ci: float = CI_LEVEL) -> tuple[float, float]:
    """Seeded bootstrap CI for the mean of paired per-item differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


@dataclass
class InterventionResult:
    law: str
    stack: str
    effect_concept: list = field(default_factory=list)
    effect_placebo: list = field(default_factory=list)
    delta_mean: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    n_positive: int = 0
    causal_positive: bool = False
    alpha: float = 0.0
    degenerate: bool = False


DEGENERATE_EPS = 1e-9


def is_intervention_degenerate(alpha: float, gaps: np.ndarray, eps: float = DEGENERATE_EPS) -> bool:
    """True iff the intervention test could not possibly have detected
    anything: either the concept direction collapsed (alpha ~ 0, so the
    concept and placebo patches are numerically identical), or the
    obey/violate observed gap is ~0 for EVERY test pair (the fixed
    HORIZON window never observes the frames where obey and violate
    actually diverge -- e.g. permanence's occluder hides the deletion for
    longer than HORIZON). A tight [0, 0] CI in either case is a vacuous
    "test did not fire", not an informative null."""
    gaps = np.asarray(gaps, dtype=np.float64)
    if alpha < eps:
        return True
    return bool(np.max(gaps) < eps) if gaps.size else True


def _random_unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def run_intervention_law_stack(
    law_name: str,
    encoder,
    predictor,
    k: int = K_CONTEXT,
    seed: int = 0,
    cache_dir: str = "cache",
) -> InterventionResult:
    """Full per-(stack, law) intervention: concept direction from
    probe-TRAIN pairs, concept+placebo patched rollouts evaluated on every
    probe-TEST pair, paired bootstrap stats over the 16 test pairs."""
    train_pairs = probe_pairs(law_name, "train")
    test_pairs = probe_pairs(law_name, "test")

    direction, alpha = concept_direction(train_pairs, encoder, cache_dir=cache_dir)
    dim = direction.shape[0]

    effect_concept: list[float] = []
    effect_placebo: list[float] = []
    gaps: list[float] = []
    for i, pair in enumerate(test_pairs):
        obey_z, violate_z = latent_frames(pair, encoder, cache_dir=cache_dir)
        cf = pair.critical_frame
        max_h = min(HORIZON, obey_z.shape[0] - cf, violate_z.shape[0] - cf)
        context = obey_z[cf - k:cf]
        obey_observed = obey_z[cf:cf + max_h]
        violate_observed = violate_z[cf:cf + max_h]
        gaps.append(dist(obey_observed, violate_observed))

        rollout_c = patched_rollout(predictor, context, direction, alpha, horizon=max_h)
        effect_concept.append(effect_score(rollout_c, obey_observed, violate_observed))

        placebo_dir = _random_unit(dim, seed=seed * 1_000_003 + i)
        rollout_p = patched_rollout(predictor, context, placebo_dir, alpha, horizon=max_h)
        effect_placebo.append(effect_score(rollout_p, obey_observed, violate_observed))

    effect_concept_arr = np.asarray(effect_concept, dtype=np.float64)
    effect_placebo_arr = np.asarray(effect_placebo, dtype=np.float64)
    diffs = effect_concept_arr - effect_placebo_arr

    delta_mean = float(diffs.mean())
    ci_lo, ci_hi = paired_bootstrap_ci(diffs, seed=seed)
    causal_positive = bool(ci_lo > 0.0 and delta_mean > 0.0)
    degenerate = is_intervention_degenerate(alpha, np.asarray(gaps, dtype=np.float64))

    return InterventionResult(
        law=law_name,
        stack=getattr(encoder, "name", str(encoder)),
        effect_concept=effect_concept_arr.tolist(),
        effect_placebo=effect_placebo_arr.tolist(),
        delta_mean=delta_mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_positive=int(np.sum(diffs > 0)),
        causal_positive=causal_positive,
        alpha=alpha,
        degenerate=degenerate,
    )
