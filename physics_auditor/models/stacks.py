"""Wires a frozen encoder + trained LatentPredictor into the ModelAdapter
protocol via closed-loop latent rollout: surprise = mean per-dim MSE between
rolled-out and observed latents over the post-critical-frame horizon."""
import numpy as np

from physics_auditor.generator.clip import Clip
from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.models.predictor import LatentPredictor


class LatentStackAdapter:
    def __init__(self, name: str, encoder, predictor: LatentPredictor, k: int = 4):
        self.name = name
        self.encoder = encoder
        self.predictor = predictor
        self.k = k

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8, cache_dir: str = "cache") -> float:
        z = encode_clip_cached(self.encoder, clip, cache_dir=cache_dir)

        max_horizon = min(horizon, len(z) - critical_frame)
        context = z[critical_frame - self.k:critical_frame]
        observed = z[critical_frame:critical_frame + max_horizon]

        predicted = self.predictor.rollout(context, max_horizon)
        return float(np.mean((predicted - observed) ** 2))
