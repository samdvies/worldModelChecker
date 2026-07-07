"""TDD: probes/mechanistic/decodability.py -- closed-form ridge decodability
probes (numpy only, no sklearn), train-stats-only z-scoring, and the real
3-stacks x 4-laws decode grid."""
import numpy as np
import pytest

from physics_auditor.probes.mechanistic.decodability import (
    DecodeResult,
    accuracy_from_scores,
    r2_score,
    ridge_fit,
    ridge_predict,
    window_is_structurally_blind,
    zscore_apply,
    zscore_fit,
)


def test_ridge_recovers_known_linear_map_on_synthetic_data():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    true_w = rng.normal(size=d)
    true_b = 1.7
    X = rng.normal(size=(n, d))
    y = X @ true_w + true_b + rng.normal(scale=1e-3, size=n)

    mean, std = zscore_fit(X)
    Xz = zscore_apply(X, mean, std)
    w = ridge_fit(Xz, y)
    pred = ridge_predict(Xz, w)

    assert r2_score(y, pred) > 0.99


def test_zscore_uses_train_stats_only_no_silent_renormalisation():
    rng = np.random.default_rng(1)
    n, d = 300, 4
    true_w = rng.normal(size=d)

    X_train = rng.normal(loc=5.0, scale=2.0, size=(n, d))  # shifted train
    y_train = X_train @ true_w

    mean, std = zscore_fit(X_train)
    Xtr_z = zscore_apply(X_train, mean, std)
    w = ridge_fit(Xtr_z, y_train)

    # Test data drawn from the SAME (unshifted-relative) distribution as
    # train, just a fresh sample -- if z-scoring is done correctly (using
    # train mean/std), predictions should still track y_test well.
    X_test = rng.normal(loc=5.0, scale=2.0, size=(100, d))
    y_test = X_test @ true_w
    Xte_z_correct = zscore_apply(X_test, mean, std)
    pred_correct = ridge_predict(Xte_z_correct, w)
    assert r2_score(y_test, pred_correct) > 0.99

    # If test were instead (incorrectly) re-normalised using ITS OWN stats
    # while X_test is unshifted (loc=0) -- simulating "silent renormalise"
    # -- the two z-scored representations must differ, proving zscore_apply
    # doesn't recompute stats from whatever array it's given.
    X_test_unshifted = rng.normal(loc=0.0, scale=2.0, size=(100, d))
    wrong_mean, wrong_std = zscore_fit(X_test_unshifted)
    Xte_wrong = zscore_apply(X_test_unshifted, wrong_mean, wrong_std)
    Xte_using_train_stats = zscore_apply(X_test_unshifted, mean, std)
    assert not np.allclose(Xte_wrong, Xte_using_train_stats)


def test_threshold_accuracy_on_synthetic_separable_data_is_one():
    rng = np.random.default_rng(2)
    n, d = 200, 3
    X = rng.normal(size=(n, d))
    X[:, 0] += np.where(X[:, 0] >= 0, 5.0, -5.0)  # push each point away from the boundary
    true_w = np.array([3.0, 0.0, 0.0])
    y_bool = (X @ true_w) > 0.0

    mean, std = zscore_fit(X)
    Xz = zscore_apply(X, mean, std)
    w = ridge_fit(Xz, y_bool.astype(np.float64))
    pred = ridge_predict(Xz, w)

    assert accuracy_from_scores(y_bool, pred) == 1.0


def test_ridge_fit_rejects_mismatched_shapes():
    X = np.zeros((10, 3))
    y = np.zeros(9)
    with pytest.raises(ValueError):
        ridge_fit(X, y)


def test_decode_result_is_a_dataclass_with_law_stack_metrics():
    result = DecodeResult(law="gravity", stack="raw-pixel", metrics={"vy": {"kind": "r2", "value": 0.9}})
    assert result.law == "gravity"
    assert result.stack == "raw-pixel"
    assert result.metrics["vy"]["value"] == 0.9


def test_decode_result_defaults_degenerate_false():
    result = DecodeResult(law="gravity", stack="raw-pixel")
    assert result.degenerate is False


# --- structural-blindness guard: permanence's occluder hides the ball for
# several frames past critical_frame, so the fixed WINDOW_OFFSETS (cf-4..
# cf+7) can be entirely pixel-identical between obey and violate -- any
# decode accuracy in that regime is a majority-class artifact, not signal,
# and must be flagged rather than reported as a plain metric. ---------------

class _FakePair:
    def __init__(self, critical_frame):
        self.critical_frame = critical_frame


def test_window_is_structurally_blind_true_when_latents_identical(monkeypatch):
    import physics_auditor.probes.mechanistic.decodability as decodability_mod

    pairs = [_FakePair(critical_frame=10), _FakePair(critical_frame=10)]
    z = np.ones((30, 4))

    def fake_latent_frames(pair, encoder, cache_dir="cache"):
        return z, z.copy()

    monkeypatch.setattr(decodability_mod, "latent_frames", fake_latent_frames)
    assert window_is_structurally_blind(pairs, encoder=None) is True


def test_window_is_structurally_blind_false_when_any_frame_diverges(monkeypatch):
    import physics_auditor.probes.mechanistic.decodability as decodability_mod

    pairs = [_FakePair(critical_frame=10), _FakePair(critical_frame=10)]
    obey_z = np.ones((30, 4))
    violate_z = obey_z.copy()
    violate_z[10, 0] = 5.0  # diverges right at the critical frame, inside the window

    def fake_latent_frames(pair, encoder, cache_dir="cache"):
        return obey_z, violate_z

    monkeypatch.setattr(decodability_mod, "latent_frames", fake_latent_frames)
    assert window_is_structurally_blind(pairs, encoder=None) is False


def test_window_is_structurally_blind_on_real_permanence_pairs():
    """Integration check against the real generator/cache: permanence's
    occluder makes the fixed decode window structurally blind."""
    from physics_auditor.models.encoders import RawPixelEncoder
    from physics_auditor.probes.mechanistic.data import probe_pairs

    enc = RawPixelEncoder()
    pairs = probe_pairs("permanence", "train", seeds=range(300, 302))
    assert window_is_structurally_blind(pairs, enc) is True


def test_window_is_structurally_blind_on_real_gravity_pairs_is_false():
    """Sanity control: gravity's divergence is visible well within the
    decode window, so the guard must not fire for it."""
    from physics_auditor.models.encoders import RawPixelEncoder
    from physics_auditor.probes.mechanistic.data import probe_pairs

    enc = RawPixelEncoder()
    pairs = probe_pairs("gravity", "train", seeds=range(300, 302))
    assert window_is_structurally_blind(pairs, enc) is False
