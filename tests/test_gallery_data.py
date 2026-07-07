"""Unit tests for the minimal-pair gallery data helpers (physics_auditor/gallery).

Covers pure trace-computation functions in isolation (no encoder/predictor
training, no monitor calibration) -- shapes, oracle spikes at the critical
frame on violate clips only, and prefix-frame identity between obey/violate
before the critical frame.
"""
import numpy as np
import pytest

from physics_auditor.gallery.traces import (
    clip_frames_base64,
    frame_to_png_base64,
    latent_stack_one_step_trace,
    normalize_trace_pair,
    oracle_one_step_trace,
    pixel_linear_one_step_trace,
    pixel_static_one_step_trace,
)
from physics_auditor.laws.gravity import GravityLaw


class _LastFramePredictor:
    """rollout(context, 1) predicts context[-1] unchanged (delta=0)."""

    def rollout(self, context: np.ndarray, n: int) -> np.ndarray:
        return np.tile(context[-1], (n, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def gravity_pair():
    return GravityLaw().generate_pair(seed=0)


def test_frame_to_png_base64_is_decodable_and_scaled():
    import base64
    import io

    from PIL import Image

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]
    b64 = frame_to_png_base64(frame, scale=3)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (12, 12)  # 4*3, 4*3
    arr = np.array(img)
    # nearest-neighbour upscale: the 3x3 block from the top-left pixel stays red.
    assert (arr[0:3, 0:3] == [255, 0, 0]).all()


def test_clip_frames_base64_length_matches_frame_count(gravity_pair):
    b64s = clip_frames_base64(gravity_pair.obey, scale=3)
    assert len(b64s) == gravity_pair.obey.frames.shape[0]
    assert all(isinstance(s, str) and s for s in b64s)


def test_prefix_frames_identical_before_critical_frame(gravity_pair):
    obey_b64 = clip_frames_base64(gravity_pair.obey, scale=3)
    violate_b64 = clip_frames_base64(gravity_pair.violate, scale=3)
    cf = gravity_pair.critical_frame
    for t in range(cf):
        assert obey_b64[t] == violate_b64[t], f"frame {t} differs before critical_frame {cf}"


def test_oracle_trace_near_zero_on_obey(gravity_pair):
    trace = oracle_one_step_trace(gravity_pair.obey)
    assert trace.shape == (gravity_pair.obey.frames.shape[0],)
    assert trace[0] == 0.0
    # lawful clip: exactly zero until the ball hits the ground/wraps and
    # pymunk's re-solved-from-state collision response introduces tiny
    # (sub-percent) numerical drift -- nowhere near the violate spike below.
    assert np.max(trace[:gravity_pair.critical_frame]) == 0.0
    assert np.max(trace) < 0.03


def test_oracle_trace_spikes_at_critical_frame_on_violate(gravity_pair):
    trace = oracle_one_step_trace(gravity_pair.violate)
    cf = gravity_pair.critical_frame
    assert trace.shape == (gravity_pair.violate.frames.shape[0],)
    # flat (exactly zero) strictly before the critical frame...
    assert np.max(trace[:cf]) == 0.0
    # ...and a real spike at-or-after it (gravity turned off -> lawful sim diverges)
    # well above the obey clip's numerical-drift noise floor.
    assert trace[cf] > 0.03


def test_pixel_static_trace_shape_and_zero_first_frame(gravity_pair):
    trace = pixel_static_one_step_trace(gravity_pair.obey)
    assert trace.shape == (gravity_pair.obey.frames.shape[0],)
    assert trace[0] == 0.0
    assert np.all(trace >= 0.0)


def test_pixel_linear_trace_shape_and_zero_first_two_frames(gravity_pair):
    trace = pixel_linear_one_step_trace(gravity_pair.obey)
    assert trace.shape == (gravity_pair.obey.frames.shape[0],)
    assert trace[0] == 0.0
    assert trace[1] == 0.0
    assert np.all(trace >= 0.0)


def test_latent_stack_trace_shape_and_zero_prefix():
    pred = _LastFramePredictor()
    z = np.random.RandomState(0).normal(size=(20, 5)).astype(np.float32)
    trace = latent_stack_one_step_trace(z, pred, k=4)
    assert trace.shape == (20,)
    assert np.allclose(trace[:4], 0.0)


def test_latent_stack_trace_zero_for_static_latents():
    """A perfectly static latent stream is predicted exactly by the
    predict-last-latent baseline -> zero error for every t >= k."""
    pred = _LastFramePredictor()
    z = np.zeros((10, 3), dtype=np.float32)
    trace = latent_stack_one_step_trace(z, pred, k=4)
    assert np.allclose(trace, 0.0)


def test_latent_stack_trace_nonzero_after_jump():
    pred = _LastFramePredictor()
    z = np.zeros((10, 3), dtype=np.float32)
    z[6:] = np.array([5.0, 0.0, 0.0])
    trace = latent_stack_one_step_trace(z, pred, k=4)
    # the jump at t=6 is unseen by the last-frame baseline at t=6 (context
    # ends at t=5, still all-zero) -> nonzero error exactly there.
    assert trace[6] > 0.0
    assert trace[7] == 0.0  # by t=7 the jump is already in context; baseline catches up


def test_normalize_trace_pair_scales_by_obey_max():
    obey = np.array([0.0, 2.0, 4.0])
    violate = np.array([0.0, 1.0, 8.0])
    norm_obey, norm_violate, normalizer = normalize_trace_pair(obey, violate)
    assert normalizer == 4.0
    assert np.allclose(norm_obey, [0.0, 0.5, 1.0])
    assert np.allclose(norm_violate, [0.0, 0.25, 2.0])


def test_normalize_trace_pair_handles_all_zero_obey():
    obey = np.zeros(5)
    violate = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    norm_obey, norm_violate, normalizer = normalize_trace_pair(obey, violate)
    assert normalizer == 1.0  # fallback, no div-by-zero
    assert np.allclose(norm_obey, obey)
    assert np.allclose(norm_violate, violate)
