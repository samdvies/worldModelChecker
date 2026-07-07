"""Common protocols for model adapters and encoders under test."""
from typing import Protocol

import numpy as np

from physics_auditor.generator.clip import Clip


class ModelAdapter(Protocol):
    name: str

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float: ...


class Encoder(Protocol):
    """A frame encoder producing a per-frame latent representation.

    cache_key MUST incorporate a hash of the encoder's weights (raw-pixel:
    static 'raw16-v1'; torch encoders: name + sha1(state_dict bytes)[:8])
    so retraining invalidates stale cached latents.
    """

    name: str
    cache_key: str

    def encode(self, clip: Clip) -> np.ndarray: ...  # (T, D) float32
