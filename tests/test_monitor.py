"""Unit tests for the thin-slice runtime monitor (monitor/detector.py) on
synthetic latents -- no encoder/cache/predictor training involved."""
import numpy as np
import pytest

from physics_auditor.monitor.detector import (
    calibrate_threshold,
    concept_scores,
    first_fire,
    frame_residuals,
)


class _LastFramePredictor:
    """Baseline predictor: rollout(context, 1) predicts context[-1] unchanged
    (delta=0), matching the untrained-predictor floor described in
    models/predictor.py."""

    def rollout(self, context: np.ndarray, n: int) -> np.ndarray:
        return np.tile(context[-1], (n, 1)).astype(np.float32)


def _clip_with_jump(k: int, dim: int, jump_frame: int, jump_vec: np.ndarray, n_frames: int = 10) -> np.ndarray:
    """Constant latents until jump_frame, then a step of `jump_vec`, held
    constant afterwards."""
    z = np.zeros((n_frames, dim), dtype=np.float32)
    z[jump_frame:] = jump_vec
    return z


def test_frame_residuals_shapes_and_zero_baseline():
    pred = _LastFramePredictor()
    z = np.zeros((10, 3), dtype=np.float32)  # perfectly static -> baseline predicts exactly
    res = frame_residuals(z, pred, k=4)
    assert res.shape == (6, 3)
    assert np.allclose(res, 0.0)


def test_frame_residuals_empty_when_too_short():
    pred = _LastFramePredictor()
    z = np.zeros((3, 3), dtype=np.float32)
    res = frame_residuals(z, pred, k=4)
    assert res.shape == (0, 3)


def test_step_change_along_direction_fires():
    """A step change along the concept direction produces a large score and
    fires; the threshold is calibrated from lawful (no-jump) clips."""
    pred = _LastFramePredictor()
    d = np.array([1.0, 0.0, 0.0])
    k = 4

    obey_clips = [np.random.RandomState(i).normal(scale=1e-4, size=(10, 3)).astype(np.float32) for i in range(5)]
    obey_scores = [concept_scores(frame_residuals(c, pred, k=k), d) for c in obey_clips]
    threshold = calibrate_threshold(obey_scores)

    violate = _clip_with_jump(k, 3, jump_frame=6, jump_vec=np.array([5.0, 0.0, 0.0]))
    scores = concept_scores(frame_residuals(violate, pred, k=k), d)
    fired = first_fire(scores, threshold, k=k)

    assert fired == 6  # residual appears at t=6 (first frame after the jump)


def test_orthogonal_step_does_not_fire_with_lambda_zero():
    """Equal-magnitude step orthogonal to d must NOT fire when lambda_s=0.0
    (pure concept projection, no magnitude term)."""
    pred = _LastFramePredictor()
    d = np.array([1.0, 0.0, 0.0])
    k = 4

    obey_clips = [np.random.RandomState(i).normal(scale=1e-4, size=(10, 3)).astype(np.float32) for i in range(5)]
    obey_scores = [concept_scores(frame_residuals(c, pred, k=k), d, lambda_s=0.0) for c in obey_clips]
    threshold = calibrate_threshold(obey_scores)

    violate = _clip_with_jump(k, 3, jump_frame=6, jump_vec=np.array([0.0, 5.0, 0.0]))
    scores = concept_scores(frame_residuals(violate, pred, k=k), d, lambda_s=0.0)
    fired = first_fire(scores, threshold, k=k)

    assert fired is None


def test_orthogonal_step_can_fire_with_lambda_nonzero():
    """The same orthogonal step DOES raise the score once lambda_s > 0 adds
    an ||r||_2 magnitude term -- confirms the combined form is available and
    behaves differently from the pure-projection form."""
    pred = _LastFramePredictor()
    d = np.array([1.0, 0.0, 0.0])
    k = 4

    obey_clips = [np.random.RandomState(i).normal(scale=1e-4, size=(10, 3)).astype(np.float32) for i in range(5)]
    obey_scores = [concept_scores(frame_residuals(c, pred, k=k), d, lambda_s=1.0) for c in obey_clips]
    threshold = calibrate_threshold(obey_scores)

    violate = _clip_with_jump(k, 3, jump_frame=6, jump_vec=np.array([0.0, 5.0, 0.0]))
    scores = concept_scores(frame_residuals(violate, pred, k=k), d, lambda_s=1.0)
    fired = first_fire(scores, threshold, k=k)

    assert fired == 6


def test_calibrate_threshold_is_99th_percentile():
    scores = [np.arange(1, 101, dtype=np.float64)]  # 1..100
    threshold = calibrate_threshold(scores, percentile=99.0)
    assert threshold == pytest.approx(np.percentile(np.arange(1, 101), 99.0))


def test_calibrate_threshold_pools_across_clips():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    pooled_expected = np.percentile(np.concatenate([a, b]), 99.0)
    assert calibrate_threshold([a, b], percentile=99.0) == pytest.approx(pooled_expected)


def test_calibrate_threshold_empty_scores_is_zero():
    assert calibrate_threshold([np.zeros(0), np.zeros(0)]) == 0.0


def test_first_fire_none_when_never_above():
    scores = np.array([0.1, 0.2, 0.15])
    assert first_fire(scores, threshold=1.0, k=4) is None


def test_first_fire_returns_absolute_frame_index():
    scores = np.array([0.0, 0.0, 2.0, 0.0])  # index 2 -> absolute frame k+2
    assert first_fire(scores, threshold=1.0, k=4) == 6
