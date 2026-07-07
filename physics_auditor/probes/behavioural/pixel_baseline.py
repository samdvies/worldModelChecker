"""Pixel-space baseline behavioural adapters: no learning, no latents.

Each predicts the next `horizon` frames from the two frames immediately
before critical_frame and scores surprise as mean per-pixel MSE (in
[0,1]-normalised pixel space) against what actually happened. These are the
floor every learned stack must beat: pixel_floor(law) = max(auroc-static,
auroc-linear).
"""
import numpy as np

from physics_auditor.generator.clip import Clip


def _norm_frames(clip: Clip) -> np.ndarray:
    return clip.frames.astype(np.float32) / 255.0


class PixelStaticAdapter:
    """Predicts the last-seen frame repeated for the whole horizon."""

    name = "pixel-static"

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float:
        frames = _norm_frames(clip)
        max_horizon = min(horizon, len(frames) - critical_frame)
        last = frames[critical_frame - 1]
        actual = frames[critical_frame:critical_frame + max_horizon]
        errors = [np.mean((last - actual[i]) ** 2) for i in range(max_horizon)]
        return float(np.mean(errors)) if errors else 0.0


class PixelLinearAdapter:
    """Predicts constant-velocity extrapolation from the last two frames."""

    name = "pixel-linear"

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float:
        frames = _norm_frames(clip)
        max_horizon = min(horizon, len(frames) - critical_frame)
        f_prev2 = frames[critical_frame - 2]
        f_prev1 = frames[critical_frame - 1]
        velocity = f_prev1 - f_prev2
        actual = frames[critical_frame:critical_frame + max_horizon]
        errors = []
        for t in range(1, max_horizon + 1):
            predicted = np.clip(f_prev1 + t * velocity, 0.0, 1.0)
            errors.append(np.mean((predicted - actual[t - 1]) ** 2))
        return float(np.mean(errors)) if errors else 0.0
