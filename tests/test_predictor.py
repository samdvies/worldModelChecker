"""TDD: models/predictor.py -- LatentPredictor learns next-latent dynamics
from a 4-latent context and can be rolled out closed-loop.

Fast synthetic-latent tests only (linear dynamical system, few epochs) --
real predictor training on cached stack latents is run separately via
scripts/train_stacks.py, not in the test suite.
"""
import numpy as np
import pytest
import torch

from physics_auditor.models.predictor import (
    LatentPredictor,
    PredictorVersionMismatch,
    load_predictor,
    save_predictor,
)


def _linear_system_latents(dim=8, t=200, seed=0):
    """z_{t+1} = A z_t (+ small noise), a linear dynamical system a small MLP
    should learn to near-zero MSE within a handful of epochs."""
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.2, size=(dim, dim)).astype(np.float32)
    z = np.zeros((t, dim), dtype=np.float32)
    z[0] = rng.normal(size=dim).astype(np.float32)
    for i in range(1, t):
        z[i] = A @ z[i - 1] + rng.normal(scale=1e-4, size=dim).astype(np.float32)
    return z


def _drifting_latents(dim=8, t=200, seed=0):
    """Smooth bounded oscillation per dimension (+ small noise): unlike a
    contractive linear system that decays to ~0 (where last-latent IS
    near-optimal), consecutive samples here differ measurably -- so the
    last-latent baseline is measurably wrong, while a network that reads the
    local trend out of its 4-latent context can extrapolate it well."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.02, 0.04, size=dim).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=dim).astype(np.float32)
    amp = rng.uniform(0.5, 1.5, size=dim).astype(np.float32)
    steps = np.arange(t, dtype=np.float32)
    z = amp[None, :] * np.sin(2 * np.pi * freq[None, :] * steps[:, None] + phase[None, :])
    z = z.astype(np.float32) + rng.normal(scale=0.01, size=z.shape).astype(np.float32)
    return z.astype(np.float32)


def test_predictor_learns_linear_dynamics_to_low_mse():
    dim = 8
    train_latents = [_linear_system_latents(dim=dim, t=200, seed=s) for s in range(3)]
    val_latents = [_linear_system_latents(dim=dim, t=60, seed=100)]

    pred = LatentPredictor(dim=dim)
    result = pred.fit(train_latents, val_latents=val_latents, epochs=30, batch_size=64, lr=1e-3, seed=0)

    assert result["val_losses"][-1] < 0.05


def test_rollout_shape_and_dtype():
    dim = 8
    pred = LatentPredictor(dim=dim)
    context = np.random.default_rng(0).normal(size=(4, dim)).astype(np.float32)
    out = pred.rollout(context, n=8)
    assert out.shape == (8, dim)
    assert out.dtype == np.float32


def test_rollout_is_deterministic():
    dim = 8
    pred = LatentPredictor(dim=dim)
    context = np.random.default_rng(0).normal(size=(4, dim)).astype(np.float32)
    out1 = pred.rollout(context, n=8)
    out2 = pred.rollout(context, n=8)
    np.testing.assert_array_equal(out1, out2)


def test_fit_reinitialises_deterministically_from_seed():
    dim = 6
    latents = [_linear_system_latents(dim=dim, t=50, seed=1)]
    pred_a = LatentPredictor(dim=dim)
    pred_a.fit(latents, epochs=2, batch_size=32, lr=1e-3, seed=0)
    pred_b = LatentPredictor(dim=dim)
    pred_b.fit(latents, epochs=2, batch_size=32, lr=1e-3, seed=0)

    sd_a = pred_a.model.state_dict()
    sd_b = pred_b.model.state_dict()
    for k in sd_a:
        torch.testing.assert_close(sd_a[k], sd_b[k])


def test_fit_without_val_returns_none_val_losses():
    dim = 4
    latents = [_linear_system_latents(dim=dim, t=30, seed=2)]
    pred = LatentPredictor(dim=dim)
    result = pred.fit(latents, epochs=2, batch_size=32, lr=1e-3, seed=0)
    assert result["val_losses"] is None
    assert len(result["train_losses"]) == 2


def _last_latent_baseline_mse(latents, k=4):
    errs = []
    for z in latents:
        t = z.shape[0]
        for i in range(t - k):
            errs.append(np.mean((z[i + k - 1] - z[i + k]) ** 2))
    return float(np.mean(errs))


def test_untrained_predictor_matches_last_latent_baseline_exactly():
    """Regression test (reviewer finding: predictors underperformed the
    trivial predict-last-latent baseline by 15x-130x): the model is
    residual/zero-init, so before any training its predictions ARE the
    baseline exactly -- training can only improve on it, never start worse."""
    dim = 8
    pred = LatentPredictor(dim=dim)
    context = np.random.default_rng(0).normal(size=(4, dim)).astype(np.float32)
    out = pred.rollout(context, n=1)
    np.testing.assert_allclose(out[0], context[-1], atol=1e-6)


def test_predictor_beats_last_latent_baseline_after_training():
    """Regression test for the same finding: on real-ish dynamics (a linear
    system perturbed by small noise, so the naive baseline is already quite
    good), a trained predictor must still beat it on held-out val data."""
    dim = 8
    train_latents = [_drifting_latents(dim=dim, t=200, seed=s) for s in range(3)]
    val_latents = [_drifting_latents(dim=dim, t=60, seed=100)]
    baseline = _last_latent_baseline_mse(val_latents, k=4)

    pred = LatentPredictor(dim=dim)
    result = pred.fit(train_latents, val_latents=val_latents, epochs=15, batch_size=256, lr=1e-3, seed=0)

    assert result["val_losses"][-1] < baseline


def test_save_and_load_predictor_round_trip(tmp_path):
    dim = 6
    pred = LatentPredictor(dim=dim)
    latents = [_linear_system_latents(dim=dim, t=30, seed=3)]
    pred.fit(latents, epochs=2, batch_size=32, lr=1e-3, seed=0)

    path = tmp_path / "predictor_test.pt"
    save_predictor(pred, path, encoder_cache_key="tiny-cnn-ae-deadbeef")

    loaded = load_predictor(path, dim=dim, encoder_cache_key="tiny-cnn-ae-deadbeef")
    sd_a = pred.model.state_dict()
    sd_b = loaded.model.state_dict()
    for k in sd_a:
        torch.testing.assert_close(sd_a[k], sd_b[k])


def test_load_predictor_raises_on_encoder_cache_key_mismatch(tmp_path):
    dim = 6
    pred = LatentPredictor(dim=dim)
    path = tmp_path / "predictor_test.pt"
    save_predictor(pred, path, encoder_cache_key="tiny-cnn-ae-oldhash01")

    with pytest.raises(PredictorVersionMismatch):
        load_predictor(path, dim=dim, encoder_cache_key="tiny-cnn-ae-newhash02")


def test_fit_never_ends_worse_than_baseline_even_on_near_static_latents():
    """Regression test: on latents that barely change frame-to-frame (as
    tiny-cnn-pred's real encoder output turned out to be), a few epochs of
    Adam noise can nudge a residual predictor's weights slightly off the
    zero-delta optimum. fit() must checkpoint on best val loss (including
    the untrained, exactly-baseline starting point) so it can never return
    a model worse than the last-latent floor."""
    dim = 8
    rng = np.random.default_rng(7)
    base = rng.normal(size=dim).astype(np.float32)
    train_latents = [base[None, :] + rng.normal(scale=1e-4, size=(120, dim)).astype(np.float32) for _ in range(2)]
    val_latents = [base[None, :] + rng.normal(scale=1e-4, size=(60, dim)).astype(np.float32)]
    baseline = _last_latent_baseline_mse(val_latents, k=4)

    pred = LatentPredictor(dim=dim)
    pred.fit(train_latents, val_latents=val_latents, epochs=15, batch_size=32, lr=1e-3, seed=0)

    # the returned model itself must reflect the best (<= baseline) checkpoint
    val_ctx, val_tgt = _windows_for_test(val_latents, k=4)
    with torch.no_grad():
        final_mse = torch.mean((pred.model(val_ctx) - val_tgt) ** 2).item()
    assert final_mse <= baseline * 1.0000001


def _windows_for_test(latents, k):
    from physics_auditor.models.predictor import _windows
    return _windows(latents, k)


def test_load_predictor_raises_if_meta_missing(tmp_path):
    dim = 6
    pred = LatentPredictor(dim=dim)
    path = tmp_path / "predictor_test.pt"
    torch.save(pred.model.state_dict(), path)  # no meta sidecar written

    with pytest.raises(PredictorVersionMismatch):
        load_predictor(path, dim=dim, encoder_cache_key="tiny-cnn-ae-anyhash0")
