"""Thin-slice runtime VoE monitor (stretch goal): a per-(stack, law) detector
that watches a clip's cached latents frame by frame and fires as soon as the
one-step-ahead prediction residual moves along the law's concept direction
more than a lawful (obey) baseline ever does.

Design (pinned):
  - k=4 context frames feed the stack's own trained LatentPredictor.
  - residual r_t = z_t - predictor.rollout(z[t-k:t], 1)[0], for t = k..T-1.
  - score_t = |proj of r_t onto unit direction d| + lambda_s * ||r_t||_2
    (lambda_s=0.0 is the primary/pinned variant -- pure concept projection;
    the combined form is kept available and evaluated as a second variant).
  - threshold = 99th percentile of ALL frame scores pooled over LAWFUL
    (obey) probe-TRAIN clips (seeds 300..331) -- never touches probe-test.
  - fire_frame(clip) = first absolute frame index t where score_t exceeds
    the calibrated threshold, else None.

This module only computes scores from latents it is handed; it does not
know about seed ranges or the probe-train/test split -- that lives in
scripts/run_monitor_eval.py, which is the only caller responsible for feeding
the correct (sacred) seed ranges into calibrate()/evaluation.
"""
from dataclasses import dataclass

import numpy as np

from physics_auditor.models.cache import encode_clip_cached

K_DEFAULT = 4
CALIBRATION_PERCENTILE = 99.0


def frame_residuals(latents: np.ndarray, predictor, k: int = K_DEFAULT) -> np.ndarray:
    """residual r_t = z_t - predictor.rollout(z[t-k:t], 1)[0] for t = k..T-1.

    latents: (T, D). Returns (max(T-k, 0), D); empty if T <= k.
    """
    t_total = latents.shape[0]
    dim = latents.shape[1] if latents.ndim == 2 else 0
    if t_total <= k:
        return np.zeros((0, dim), dtype=latents.dtype)
    residuals = []
    for t in range(k, t_total):
        context = latents[t - k:t]
        pred = predictor.rollout(context, 1)[0]
        residuals.append(latents[t] - pred)
    return np.stack(residuals)


def concept_scores(residuals: np.ndarray, direction: np.ndarray, lambda_s: float = 0.0) -> np.ndarray:
    """score_t = |r_t . d| + lambda_s * ||r_t||_2. `direction` is assumed unit
    norm (as returned by probes.mechanistic.intervention.concept_direction)."""
    if residuals.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    proj = np.abs(residuals @ direction)
    if lambda_s == 0.0:
        return proj
    mag = np.linalg.norm(residuals, axis=1)
    return proj + lambda_s * mag


def calibrate_threshold(all_obey_scores: list[np.ndarray], percentile: float = CALIBRATION_PERCENTILE) -> float:
    """99th-percentile threshold pooled over every frame score from every
    lawful (obey) calibration clip. 0.0 if there is nothing to pool (no
    frames reached k in any clip)."""
    non_empty = [s for s in all_obey_scores if s.size]
    if not non_empty:
        return 0.0
    pooled = np.concatenate(non_empty)
    return float(np.percentile(pooled, percentile))


def first_fire(scores: np.ndarray, threshold: float, k: int = K_DEFAULT) -> int | None:
    """First absolute frame index t (t = k + i) where scores[i] > threshold,
    else None. Strictly greater-than, matching a 99th-percentile threshold
    calibrated on obey clips (obey scores at the threshold itself should not
    self-trigger)."""
    if scores.size == 0:
        return None
    above = np.flatnonzero(scores > threshold)
    if above.size == 0:
        return None
    return k + int(above[0])


@dataclass
class LawMonitor:
    """stack_adapter here is the frozen Encoder (has .encode/.cache_key via
    the shared on-disk cache) -- NOT the ModelAdapter/LatentStackAdapter
    wrapper, since the monitor needs raw per-frame latents, not a single
    scalar surprise. `predictor` is that stack's own trained LatentPredictor."""
    encoder: object
    predictor: object
    direction: np.ndarray
    law_name: str
    k: int = K_DEFAULT
    lambda_s: float = 0.0
    threshold: float | None = None

    def clip_scores(self, clip, cache_dir: str = "cache") -> np.ndarray:
        z = encode_clip_cached(self.encoder, clip, cache_dir=cache_dir)
        residuals = frame_residuals(z, self.predictor, k=self.k)
        return concept_scores(residuals, self.direction, lambda_s=self.lambda_s)

    def calibrate(self, obey_clips: list, cache_dir: str = "cache", percentile: float = CALIBRATION_PERCENTILE) -> float:
        """Sets (and returns) self.threshold from the pooled frame scores of
        `obey_clips` -- callers MUST pass only lawful (obey) probe-TRAIN
        clips (seeds 300..331); this function itself enforces nothing about
        seed provenance."""
        all_scores = [self.clip_scores(clip, cache_dir=cache_dir) for clip in obey_clips]
        self.threshold = calibrate_threshold(all_scores, percentile=percentile)
        return self.threshold

    def fire_frame(self, clip, cache_dir: str = "cache") -> int | None:
        if self.threshold is None:
            raise ValueError("LawMonitor.calibrate() must be called before fire_frame()")
        scores = self.clip_scores(clip, cache_dir=cache_dir)
        return first_fire(scores, self.threshold, k=self.k)

    def clip_score_max(self, clip, cache_dir: str = "cache") -> float:
        """Max post-k frame score for a clip -- used as the clip-level
        detection score for AUROC (violate vs obey)."""
        scores = self.clip_scores(clip, cache_dir=cache_dir)
        return float(np.max(scores)) if scores.size else 0.0
