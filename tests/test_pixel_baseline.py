"""TDD: pixel-baseline behavioural adapters ('pixel-static', 'pixel-linear').

Uses constructed frame arrays (not the engine) so the surprise math is
exercised directly against known ground truth.
"""
from dataclasses import dataclass

import numpy as np

from physics_auditor.probes.behavioural.pixel_baseline import (
    PixelLinearAdapter,
    PixelStaticAdapter,
)

SIZE = 16
SIGMA = 1.5


def _square_frame(x: int, y: int, size: int = SIZE, sq: int = 3) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[y:y + sq, x:x + sq] = 255
    return frame


def _blob_frame(cx: float, cy: float, size: int = SIZE) -> np.ndarray:
    """A soft (anti-aliased) moving blob. Hard-edged translating squares are
    NOT well predicted by pixel-space linear extrapolation (no motion
    compensation), so this is the honest constant-velocity fixture: a smooth
    intensity profile whose value at any fixed pixel varies near-affinely as
    it slides through, which is exactly what constant-velocity extrapolation
    in pixel space can track."""
    ys, xs = np.mgrid[0:size, 0:size]
    v = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA ** 2))
    u8 = (v * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


@dataclass
class _FakeClip:
    frames: np.ndarray


def _moving_blob_clip(n=14, vx=0.5, cy=8.0, start_cx=4.0):
    frames = np.stack([_blob_frame(start_cx + vx * t, cy) for t in range(n)])
    return _FakeClip(frames=frames)


def _static_clip(n=12):
    frames = np.stack([_square_frame(3, 3) for _ in range(n)])
    return _FakeClip(frames=frames)


def _sudden_appearance_clip(n=12, critical_frame=6):
    frames = []
    for t in range(n):
        if t < critical_frame:
            frames.append(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
        else:
            frames.append(_square_frame(3, 3))
    return _FakeClip(frames=np.stack(frames))


def test_names():
    assert PixelStaticAdapter().name == "pixel-static"
    assert PixelLinearAdapter().name == "pixel-linear"


def test_constant_velocity_motion_low_surprise_for_linear_high_for_static():
    clip = _moving_blob_clip()
    cf = 6
    static_surprise = PixelStaticAdapter().surprise(clip, cf, horizon=4)
    linear_surprise = PixelLinearAdapter().surprise(clip, cf, horizon=4)
    assert linear_surprise < 0.005
    assert static_surprise > 0.008
    assert linear_surprise < static_surprise


def test_static_scene_both_adapters_near_zero_surprise():
    clip = _static_clip()
    cf = 6
    assert PixelStaticAdapter().surprise(clip, cf, horizon=4) < 1e-6
    assert PixelLinearAdapter().surprise(clip, cf, horizon=4) < 1e-6


def test_sudden_appearance_both_adapters_large_surprise():
    clip = _sudden_appearance_clip(n=14, critical_frame=6)
    cf = 6
    static_surprise = PixelStaticAdapter().surprise(clip, cf, horizon=4)
    linear_surprise = PixelLinearAdapter().surprise(clip, cf, horizon=4)
    assert static_surprise > 0.01
    assert linear_surprise > 0.01


def test_surprise_clipped_horizon_when_clip_shorter_than_horizon():
    clip = _static_clip(n=8)
    cf = 6  # only 2 frames remain after cf
    # should not raise, and should still be near zero for a static scene
    assert PixelStaticAdapter().surprise(clip, cf, horizon=8) < 1e-6
