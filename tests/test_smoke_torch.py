"""Torch-backed smoke tier (pytest -m smoke_torch): the counterpart to
tests/test_smoke.py's <=10s torch-free `smoke` tier.

`import torch` alone measures ~15s cold on this machine (see
docs/failure-sweeps.md), which makes any tier that needs the real
torch-backed encoders/predictor structurally incompatible with a <=10s
budget in this environment. Rather than miss that hard rule silently (or
weaken the fast tier down to "imports only"), the torch-dependent wiring
checks live here instead, run right after `-m smoke` and before the full
suite / GPU-box dispatch -- see scripts/aws/remote_job.sh and
scripts/fresh_clone_smoke.py.

Not held to a time budget, but still fast in absolute terms (mock-injected
models only, no network/`.load()` calls -- those are GPU-box-only).
"""
import importlib
import pkgutil

import numpy as np
import pytest
import torch
import torch.nn as nn

import physics_auditor
import scripts as scripts_pkg
from tests.test_smoke import TORCH_MODULES, _tiny_config

pytestmark = pytest.mark.smoke_torch


def _submodule_names(pkg) -> list[str]:
    return [m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}.")]


@pytest.mark.parametrize("mod_name", sorted(TORCH_MODULES))
def test_torch_module_imports(mod_name):
    """Every torch-dependent physics_auditor.*/scripts.* module must import
    cleanly (the class-A/discovery counterpart to test_smoke.py's fast-tier
    import sweep, for exactly the modules that tier excludes)."""
    importlib.import_module(mod_name)


def test_raw_pixel_encoder_constructs_and_encodes():
    from physics_auditor.generator.simulate import simulate
    from physics_auditor.models.encoders import RawPixelEncoder

    clip = simulate(_tiny_config(seed=1, violate=False))
    enc = RawPixelEncoder()
    z = enc.encode(clip)
    assert z.shape == (8, 256)
    assert z.dtype == np.float32


def test_tiny_cnn_ae_and_pred_construct_and_encode():
    from physics_auditor.generator.simulate import simulate
    from physics_auditor.models.encoders import TinyCNNAE, TinyCNNPred

    clip = simulate(_tiny_config(seed=2, violate=False))
    for cls in (TinyCNNAE, TinyCNNPred):
        enc = cls()
        z = enc.encode(clip)
        assert z.shape == (8, 128)
        assert z.dtype == np.float32
        assert enc.cache_key.startswith(enc.name)


class _MockDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 384)

    def forward(self, x):
        return self.proj(x.mean(dim=[2, 3]))


class _MockVJEPA(nn.Module):
    def __init__(self, hidden_dim=1024, num_spatial=4):
        super().__init__()
        self.num_spatial = num_spatial
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, pixel_values_videos, **kwargs):
        b, t, c, h, w = pixel_values_videos.shape
        num_temporal = max(t // 2, 1)
        pooled = pixel_values_videos.mean(dim=[3, 4])[0]
        temporal_pooled = pooled[: num_temporal * 2 : 2]
        tokens = self.proj(temporal_pooled).repeat_interleave(self.num_spatial, dim=0)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = tokens.unsqueeze(0)
        return out


def test_pretrained_encoders_construct_and_mock_encode():
    """DINOv2Encoder + VJEPA2Encoder: mock-injected models only, .load() is
    never called (that hits the network and is GPU-box only)."""
    from physics_auditor.generator.simulate import simulate
    from physics_auditor.models.pretrained import DINOv2Encoder, VJEPA2Encoder

    clip = simulate(_tiny_config(seed=3, violate=False))

    dino = DINOv2Encoder(model=_MockDino())
    z_dino = dino.encode(clip)
    assert z_dino.shape == (8, 384)

    vjepa = VJEPA2Encoder(model=_MockVJEPA())
    z_vjepa = vjepa.encode(clip)
    assert z_vjepa.shape == (8, 1024)


def test_predictor_rollout_shape():
    from physics_auditor.models.predictor import LatentPredictor

    dim, k = 8, 4
    pred = LatentPredictor(dim=dim, k=k)
    context = np.zeros((k, dim), dtype=np.float32)
    out = pred.rollout(context, n=3)
    assert out.shape == (3, dim)
    assert out.dtype == np.float32
