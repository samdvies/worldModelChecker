"""TDD: models/stacks.py -- LatentStackAdapter wires an encoder + trained
LatentPredictor into the ModelAdapter.surprise(clip, critical_frame, horizon)
protocol via closed-loop latent rollout, using the on-disk latent cache.
"""
import numpy as np
import pytest

from physics_auditor.generator.dataset import eval_pairs
from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.models.encoders import RawPixelEncoder
from physics_auditor.models.predictor import LatentPredictor
from physics_auditor.models.stacks import LatentStackAdapter


class _ZeroEncoder:
    """Encoder stub producing constant zero latents, so adapter surprise math
    is exercised directly against a known-zero predictor output."""

    name = "zero-enc"
    cache_key = "zero-enc-v1"

    def __init__(self, dim=4):
        self.dim = dim

    def encode(self, clip):
        return np.zeros((len(clip.frames), self.dim), dtype=np.float32)


class _FakeConfig:
    scenario_id = "deadbeef00"


class _FakeClip:
    def __init__(self, t=20, dim=4):
        self.frames = np.zeros((t, 64, 64, 3), dtype=np.uint8)
        self.config = _FakeConfig()


def test_name_defaults_to_given_name():
    enc = _ZeroEncoder()
    pred = LatentPredictor(dim=4)
    adapter = LatentStackAdapter(name="zero-stack", encoder=enc, predictor=pred)
    assert adapter.name == "zero-stack"


def test_adapter_uses_cache_and_returns_nonnegative_float(tmp_path):
    enc = _ZeroEncoder(dim=4)
    pred = LatentPredictor(dim=4)
    adapter = LatentStackAdapter(name="zero-stack", encoder=enc, predictor=pred, k=4)
    clip = _FakeClip(t=20, dim=4)
    surprise = adapter.surprise(clip, critical_frame=8, horizon=8, cache_dir=str(tmp_path))
    assert isinstance(surprise, float)
    assert surprise >= 0.0
    # cached latents were written under the encoder's cache_key
    cached = tmp_path / enc.cache_key / clip.config.scenario_id
    assert (cached.with_suffix(".npy")).exists()


def test_adapter_surprise_runs_on_real_cached_pair():
    """Smoke test against a real cached eval pair using the raw-pixel stack
    (already populated in cache/raw16-v1) and a lightly-fit predictor."""
    enc = RawPixelEncoder()
    pairs = eval_pairs("support", seeds=range(0, 2))
    pair = pairs[0]

    z_obey = encode_clip_cached(enc, pair.obey)
    dim = z_obey.shape[1]

    pred = LatentPredictor(dim=dim)
    pred.fit([z_obey], epochs=1, batch_size=64, lr=1e-3, seed=0)

    adapter = LatentStackAdapter(name="raw-pixel", encoder=enc, predictor=pred, k=4)
    s_obey = adapter.surprise(pair.obey, pair.critical_frame, horizon=8)
    s_violate = adapter.surprise(pair.violate, pair.critical_frame, horizon=8)

    assert isinstance(s_obey, float)
    assert isinstance(s_violate, float)
    assert s_obey >= 0.0
    assert s_violate >= 0.0
