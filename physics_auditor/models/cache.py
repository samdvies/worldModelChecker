"""On-disk latent cache: encode_clip_cached(encoder, clip) memoizes by
<cache_dir>/<encoder.cache_key>/<scenario_id>.npy."""
import os

import numpy as np

from physics_auditor.generator.clip import Clip
from physics_auditor.models.base import Encoder


def encode_clip_cached(encoder: Encoder, clip: Clip, cache_dir: str = "cache") -> np.ndarray:
    key_dir = os.path.join(cache_dir, encoder.cache_key)
    path = os.path.join(key_dir, f"{clip.config.scenario_id}.npy")

    if os.path.exists(path):
        return np.load(path)

    latents = encoder.encode(clip).astype(np.float32)
    os.makedirs(key_dir, exist_ok=True)
    np.save(path, latents)
    return latents
