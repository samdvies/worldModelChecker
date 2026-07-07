"""Per-frame trace + frame-encoding helpers for the static HTML gallery
(docs/gallery-design.md, ladder step 2).

All functions here are pure given their inputs (a Clip, an array of cached
latents, a trained predictor) -- no encoder loading, no cache I/O, no
monitor calibration. That lives in scripts/build_gallery_data.py, which is
the only caller responsible for wiring real encoders/predictors/monitors and
the sacred probe-train/test seed split.
"""
import base64
import io

import numpy as np
from PIL import Image

from physics_auditor.generator.clip import Clip
from physics_auditor.generator.engine import Engine
from physics_auditor.generator.state import state_distance
from physics_auditor.monitor.detector import frame_residuals


def frame_to_png_base64(frame: np.ndarray, scale: int = 3) -> str:
    """Nearest-neighbour upscale a single (H, W, 3) uint8 frame by `scale`
    and return it as a base64-encoded PNG (no data: prefix)."""
    img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clip_frames_base64(clip: Clip, scale: int = 3) -> list[str]:
    """base64 PNG per frame, in order, upscaled by `scale`."""
    return [frame_to_png_base64(f, scale=scale) for f in clip.frames]


def oracle_one_step_trace(clip: Clip) -> np.ndarray:
    """One-step-ahead oracle surprise per frame: trace[0] = 0.0; for t >= 1,
    trace[t] = state_distance(Engine.from_state(states[t-1]).step(),
    states[t]) -- i.e. how far the actual state[t] is from what a fully
    lawful simulator would have produced from state[t-1]. Spikes exactly at
    the frame a violation first changes the trajectory."""
    states = clip.states
    t_total = len(states)
    trace = np.zeros(t_total, dtype=np.float64)
    for t in range(1, t_total):
        engine = Engine.from_state(states[t - 1])
        predicted = engine.step()
        trace[t] = state_distance(predicted, states[t])
    return trace


def pixel_static_one_step_trace(clip: Clip) -> np.ndarray:
    """trace[0] = 0.0; trace[t] = mean per-pixel MSE ([0,1]-normalised)
    between frame[t-1] (predicted, repeated) and frame[t] (actual)."""
    frames = clip.frames.astype(np.float32) / 255.0
    t_total = frames.shape[0]
    trace = np.zeros(t_total, dtype=np.float64)
    for t in range(1, t_total):
        trace[t] = float(np.mean((frames[t - 1] - frames[t]) ** 2))
    return trace


def pixel_linear_one_step_trace(clip: Clip) -> np.ndarray:
    """trace[0] = trace[1] = 0.0; trace[t] (t >= 2) = mean per-pixel MSE
    between a constant-velocity extrapolation from frames[t-2], frames[t-1]
    and the actual frame[t]."""
    frames = clip.frames.astype(np.float32) / 255.0
    t_total = frames.shape[0]
    trace = np.zeros(t_total, dtype=np.float64)
    for t in range(2, t_total):
        velocity = frames[t - 1] - frames[t - 2]
        predicted = np.clip(frames[t - 1] + velocity, 0.0, 1.0)
        trace[t] = float(np.mean((predicted - frames[t]) ** 2))
    return trace


def latent_stack_one_step_trace(latents: np.ndarray, predictor, k: int = 4) -> np.ndarray:
    """trace[t] = 0.0 for t < k; for t >= k, trace[t] = mean per-dim MSE
    between predictor's one-step teacher-forced prediction from
    latents[t-k:t] and the observed latents[t] (same residual definition as
    monitor/detector.py's frame_residuals, reduced to a scalar per frame)."""
    t_total = latents.shape[0]
    trace = np.zeros(t_total, dtype=np.float64)
    residuals = frame_residuals(latents, predictor, k=k)  # (T-k, D)
    for i, r in enumerate(residuals):
        trace[k + i] = float(np.mean(r ** 2))
    return trace


def normalize_trace_pair(obey_trace: np.ndarray, violate_trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale both traces by the obey clip's own max (so a model's obey and
    violate traces for one law are comparable on a single axis). Falls back
    to a normalizer of 1.0 (no-op) if the obey trace is all-zero, to avoid
    div-by-zero -- this itself is a signal worth surfacing (a model with
    zero obey-trace variance has nothing to normalise against)."""
    normalizer = float(np.max(obey_trace)) if obey_trace.size else 0.0
    if normalizer <= 0.0:
        normalizer = 1.0
    return obey_trace / normalizer, violate_trace / normalizer, normalizer
