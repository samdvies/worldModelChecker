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


# --- registry completeness guard: every law in ALL_LAWS (including the two
# eval-only laws, permanence-ext and support-hard) must have an entry in both
# LAW_VARIABLES and PRIMARY_VARIABLE, and PRIMARY_VARIABLE must name a
# variable that's actually present in that law's LAW_VARIABLES list. This is
# the standing guard that would have caught the KeyError('permanence-ext')
# bug in scripts/run_mechanistic.py at commit time. -------------------------

def test_every_law_has_law_variables_and_primary_variable_entries():
    from physics_auditor.laws import ALL_LAWS
    from physics_auditor.probes.mechanistic.decodability import LAW_VARIABLES, PRIMARY_VARIABLE

    for law_name in ALL_LAWS:
        assert law_name in LAW_VARIABLES, f"{law_name} missing from LAW_VARIABLES"
        assert law_name in PRIMARY_VARIABLE, f"{law_name} missing from PRIMARY_VARIABLE"


def test_primary_variable_names_a_variable_in_its_laws_variable_list():
    from physics_auditor.laws import ALL_LAWS
    from physics_auditor.probes.mechanistic.decodability import LAW_VARIABLES, PRIMARY_VARIABLE

    for law_name in ALL_LAWS:
        var_names = {v for v, _, _ in LAW_VARIABLES[law_name]}
        assert PRIMARY_VARIABLE[law_name] in var_names, (
            f"{law_name}: PRIMARY_VARIABLE {PRIMARY_VARIABLE[law_name]!r} not in {var_names}"
        )


def test_permanence_ext_reuses_permanence_label_and_primary_variable():
    from physics_auditor.probes.mechanistic import labels as L
    from physics_auditor.probes.mechanistic.decodability import LAW_VARIABLES, PRIMARY_VARIABLE

    assert LAW_VARIABLES["permanence-ext"] == [("permanence", "acc", L.permanence_label)]
    assert PRIMARY_VARIABLE["permanence-ext"] == "permanence"


def test_support_hard_reuses_support_label_and_primary_variable():
    from physics_auditor.probes.mechanistic import labels as L
    from physics_auditor.probes.mechanistic.decodability import LAW_VARIABLES, PRIMARY_VARIABLE

    assert LAW_VARIABLES["support-hard"] == [("support", "acc", L.support_label)]
    assert PRIMARY_VARIABLE["support-hard"] == "support"


# --- label-function validity on real generated pairs: confirms the two
# reused label functions actually apply cleanly to the new laws' clips. ----

def test_permanence_label_correct_on_permanence_ext_pairs():
    from physics_auditor.laws.permanence_ext import PermanenceExtLaw
    from physics_auditor.probes.mechanistic import labels as L

    for seed in (0, 1):
        pair = PermanenceExtLaw().generate_pair(seed)
        cf = pair.critical_frame
        for t in range(len(pair.obey.states)):
            assert L.permanence_label(pair.obey.states[t]) is True, f"seed {seed} obey frame {t}"
        for t in range(cf, len(pair.violate.states)):
            assert L.permanence_label(pair.violate.states[t]) is False, f"seed {seed} violate frame {t}"


def test_support_label_correct_before_critical_frame_on_support_hard_pairs():
    """Tower A is intact (block2 supported) in both branches strictly before
    critical_frame -- they're prefix-identical there by minimal-pair
    discipline. Frame 0 itself has no contact arbiters yet (the physics
    engine only populates support_edges after the first step, true for
    every law, not a support-hard quirk), so frame 1 is used instead."""
    from physics_auditor.laws.support_hard import SupportHardLaw
    from physics_auditor.probes.mechanistic import labels as L

    for seed in (0, 1):
        pair = SupportHardLaw().generate_pair(seed)
        assert L.support_label(pair.obey.states[1]) is True, f"seed {seed} obey frame 1"
        assert L.support_label(pair.violate.states[1]) is True, f"seed {seed} violate frame 1"
