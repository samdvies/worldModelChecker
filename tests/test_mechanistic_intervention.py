"""TDD: probes/mechanistic/intervention.py -- concept-direction patching
with a random-direction placebo control, on a SYNTHETIC linear system where
the concept direction and predictor are known by construction, plus a
from-scratch bootstrap-CI coverage check. numpy only."""
import numpy as np
import pytest

from physics_auditor.probes.mechanistic.intervention import (
    InterventionResult,
    _direction_from_diffs,
    _random_unit,
    dist,
    effect_score,
    is_intervention_degenerate,
    paired_bootstrap_ci,
    patched_rollout,
)


class _IdentityPredictor:
    """A trivial 'true linear map' predictor for a static synthetic system:
    the underlying state doesn't evolve, so the correct next-latent
    prediction is always the last context latent, repeated for as long as
    asked."""

    def rollout(self, context: np.ndarray, n: int) -> np.ndarray:
        last = context[-1]
        return np.tile(last, (n, 1)).astype(np.float32)


def test_dist_is_mean_per_frame_l2():
    a = np.zeros((3, 4))
    b = np.zeros((3, 4))
    b[:, 0] = 3.0
    b[:, 1] = 4.0  # each frame differs by a 3-4-5 triangle -> L2 = 5
    assert dist(a, b) == pytest.approx(5.0)


def test_direction_from_diffs_recovers_known_direction_and_alpha():
    rng = np.random.default_rng(0)
    true_v = rng.normal(size=10)
    # Several noisy observations of the same underlying diff-of-means.
    diffs = np.stack([true_v + rng.normal(scale=1e-4, size=10) for _ in range(20)])

    unit, alpha = _direction_from_diffs(diffs)

    assert alpha == pytest.approx(np.linalg.norm(true_v), rel=1e-2)
    assert np.linalg.norm(unit) == pytest.approx(1.0)
    cos_sim = unit @ (true_v / np.linalg.norm(true_v))
    assert cos_sim > 0.999


def test_synthetic_static_system_concept_effect_is_exactly_one():
    """obey is latent-zero, violate is a fixed non-zero vector v; an
    identity ('last-latent') predictor plus a concept patch of the true
    direction must reproduce the counterfactual (violate) exactly."""
    d = 12
    rng = np.random.default_rng(1)
    v = rng.normal(size=d) * 2.0
    h = 8

    obey_observed = np.zeros((h, d))
    violate_observed = np.tile(v, (h, 1))
    context = np.zeros((4, d))

    diffs = np.tile(v, (3, 1))  # cf+1..cf+3, all equal to v (static system)
    direction, alpha = _direction_from_diffs(diffs)

    predictor = _IdentityPredictor()
    rollout = patched_rollout(predictor, context, direction, alpha, horizon=h)

    assert rollout == pytest.approx(violate_observed, abs=1e-4)
    effect = effect_score(rollout, obey_observed, violate_observed)
    assert effect == pytest.approx(1.0, abs=1e-6)


def test_synthetic_placebo_effect_is_much_smaller_than_concept_effect():
    """Over many independently-seeded random placebo directions of the same
    magnitude, the placebo's effect score should average out far below the
    concept direction's effect (which is always exactly 1 by construction)
    -- the diagnostic must clearly separate 'true concept' from 'random
    direction, same size'."""
    d = 32
    rng = np.random.default_rng(2)
    v = rng.normal(size=d) * 2.0
    h = 8
    predictor = _IdentityPredictor()

    obey_observed = np.zeros((h, d))
    violate_observed = np.tile(v, (h, 1))
    context = np.zeros((4, d))

    diffs = np.tile(v, (3, 1))
    direction, alpha = _direction_from_diffs(diffs)

    n_trials = 30
    effect_concept = []
    effect_placebo = []
    for i in range(n_trials):
        rollout_c = patched_rollout(predictor, context, direction, alpha, horizon=h)
        effect_concept.append(effect_score(rollout_c, obey_observed, violate_observed))

        placebo_dir = _random_unit(d, seed=1000 + i)
        rollout_p = patched_rollout(predictor, context, placebo_dir, alpha, horizon=h)
        effect_placebo.append(effect_score(rollout_p, obey_observed, violate_observed))

    effect_concept = np.asarray(effect_concept)
    effect_placebo = np.asarray(effect_placebo)

    assert effect_concept.mean() == pytest.approx(1.0, abs=1e-6)
    # Placebo should not show the same directed, positive signal.
    assert effect_placebo.mean() < 0.3
    assert effect_concept.mean() - effect_placebo.mean() > 0.7

    diffs_paired = effect_concept - effect_placebo
    lo, hi = paired_bootstrap_ci(diffs_paired, n_resamples=2000, seed=0)
    assert lo > 0.0  # CI excludes zero
    assert lo <= hi


def test_random_unit_is_deterministic_and_unit_norm():
    a = _random_unit(16, seed=7)
    b = _random_unit(16, seed=7)
    c = _random_unit(16, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_bootstrap_ci_covers_true_mean_at_roughly_nominal_rate():
    """Classic bootstrap-CI sanity check: draw many independent synthetic
    samples from a known-mean normal distribution, build a 95% CI for each,
    and check the true mean falls inside it close to 95% of the time."""
    true_mean = 0.5
    n_trials = 200
    n_per_sample = 30
    covered = 0
    master_rng = np.random.default_rng(42)
    for trial in range(n_trials):
        sample = master_rng.normal(loc=true_mean, scale=1.0, size=n_per_sample)
        lo, hi = paired_bootstrap_ci(sample, n_resamples=1000, seed=trial)
        if lo <= true_mean <= hi:
            covered += 1
    coverage = covered / n_trials
    assert 0.85 <= coverage <= 1.0


def test_bootstrap_ci_lo_le_hi_and_reproducible_given_seed():
    rng = np.random.default_rng(3)
    diffs = rng.normal(loc=0.2, scale=0.5, size=16)
    lo1, hi1 = paired_bootstrap_ci(diffs, n_resamples=500, seed=5)
    lo2, hi2 = paired_bootstrap_ci(diffs, n_resamples=500, seed=5)
    assert lo1 == lo2 and hi1 == hi2
    assert lo1 <= hi1


def test_intervention_result_is_a_dataclass():
    result = InterventionResult(law="gravity", stack="raw-pixel", delta_mean=0.4, ci_lo=0.1, ci_hi=0.7, causal_positive=True)
    assert result.law == "gravity"
    assert result.causal_positive is True


def test_intervention_result_defaults_alpha_and_degenerate():
    result = InterventionResult(law="gravity", stack="raw-pixel")
    assert result.alpha == 0.0
    assert result.degenerate is False


# --- degeneracy guard: a fixed HORIZON/CONCEPT window can, for a law whose
# occluder hides the divergence for longer than the window, never observe
# any obey/violate gap -- alpha collapses to 0 and every effect_score is a
# structural 0/0 rather than an informative null. This must be flagged, not
# silently reported as a confident CI. ---------------------------------

def test_is_intervention_degenerate_true_when_alpha_near_zero():
    gaps = np.array([0.5, 0.6, 0.4])
    assert is_intervention_degenerate(alpha=1e-12, gaps=gaps) is True


def test_is_intervention_degenerate_true_when_all_gaps_near_zero():
    assert is_intervention_degenerate(alpha=2.0, gaps=np.zeros(16)) is True


def test_is_intervention_degenerate_false_for_healthy_case():
    gaps = np.array([0.3, 0.4, 0.5])
    assert is_intervention_degenerate(alpha=1.5, gaps=gaps) is False


def test_is_intervention_degenerate_false_when_only_some_gaps_are_zero():
    # A handful of test pairs happening to have a small gap shouldn't trip
    # the guard -- only a uniformly-zero window (structural blindness)
    # should.
    gaps = np.array([0.0, 0.0, 0.4, 0.5])
    assert is_intervention_degenerate(alpha=1.5, gaps=gaps) is False
