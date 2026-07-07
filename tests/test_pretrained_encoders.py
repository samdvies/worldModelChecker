"""TDD: DINOv2Encoder + VJEPA2Encoder pretrained adapters -- mock-injected
nn.Modules only. No weight downloads: .load() (the only place that would
hit torch.hub / HuggingFace) is never called in this suite."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.models.pretrained import DINOv2Encoder, VJEPA2Encoder


@dataclass
class _FakeConfig:
    scenario_id: str = "deadbeef"


class _FakeClip:
    def __init__(self, frames: np.ndarray, scenario_id: str = "deadbeef"):
        self.frames = frames
        self.config = _FakeConfig(scenario_id)


def _fake_clip(t=5, seed=0):
    frames = np.random.default_rng(seed).integers(0, 256, size=(t, 64, 64, 3), dtype=np.uint8)
    return _FakeClip(frames)


def _solid_clip(t, color: tuple[int, int, int]):
    frames = np.empty((t, 64, 64, 3), dtype=np.uint8)
    frames[:] = np.array(color, dtype=np.uint8)
    return _FakeClip(frames)


# ------------------------------------------------------------- DINOv2Encoder

class _MockDino(nn.Module):
    """Identity-ish mock: global-avg-pool spatial dims, then a fixed linear
    projection to 384-d -- lets us hand-compute the expected output."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=[2, 3])  # (B, 3)
        return self.proj(pooled)


def _mock_dino(seed=0) -> _MockDino:
    torch.manual_seed(seed)
    m = _MockDino()
    m.eval()
    return m


def test_dinov2_name():
    enc = DINOv2Encoder(model=_mock_dino())
    assert enc.name == "dinov2-s14"


def test_dinov2_encode_shape_and_dtype():
    enc = DINOv2Encoder(model=_mock_dino())
    clip = _fake_clip(t=6)
    z = enc.encode(clip)
    assert z.shape == (6, 384)
    assert z.dtype == np.float32


def test_dinov2_resize_normalize_pipeline_correctness():
    """A solid-colour clip survives bicubic resize to 224x224 unchanged (all
    input pixels equal => no interpolation artefacts), so the expected
    ImageNet-normalised pooled vector is exactly computable by hand."""
    model = _mock_dino(seed=1)
    enc = DINOv2Encoder(model=model)
    color = (10, 200, 80)
    clip = _solid_clip(t=2, color=color)

    z = enc.encode(clip)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalised = (np.array(color, dtype=np.float32) / 255.0 - mean) / std
    expected_pooled = torch.from_numpy(normalised).unsqueeze(0)  # (1,3)
    with torch.no_grad():
        expected = model.proj(expected_pooled).numpy()[0]

    for t in range(2):
        np.testing.assert_allclose(z[t], expected, rtol=1e-4, atol=1e-5)


def test_dinov2_cache_key_format():
    enc = DINOv2Encoder(model=_mock_dino())
    assert enc.cache_key.startswith("dinov2-s14-")
    assert len(enc.cache_key) == len("dinov2-s14-") + 8


def test_dinov2_cache_key_deterministic_same_weights():
    enc1 = DINOv2Encoder(model=_mock_dino(seed=7))
    enc2 = DINOv2Encoder(model=_mock_dino(seed=7))
    assert enc1.cache_key == enc2.cache_key


def test_dinov2_cache_key_sensitive_to_weights():
    enc1 = DINOv2Encoder(model=_mock_dino(seed=7))
    enc2 = DINOv2Encoder(model=_mock_dino(seed=8))
    assert enc1.cache_key != enc2.cache_key


def test_dinov2_requires_model_before_use():
    enc = DINOv2Encoder()
    try:
        enc.encode(_fake_clip())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    try:
        _ = enc.cache_key
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_dinov2_encode_clip_cached_round_trip(tmp_path):
    enc = DINOv2Encoder(model=_mock_dino())
    clip = _fake_clip(t=4)
    z1 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    expected = tmp_path / enc.cache_key / "deadbeef.npy"
    assert expected.exists()
    z2 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    np.testing.assert_array_equal(z1, z2)


# ------------------------------------------------------------ VJEPA2Encoder

class _MockVJEPA(nn.Module):
    """Deterministic mock exposing the same call signature as the real HF
    model (pixel_values=... -> object with .last_hidden_state)."""

    def __init__(self, hidden_dim=1024, num_spatial=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_spatial = num_spatial
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, pixel_values: torch.Tensor):
        b, t, c, h, w = pixel_values.shape
        num_temporal = max(t // 2, 1)
        pooled = pixel_values.mean(dim=[3, 4])[0]  # (t, 3)
        # one token vector per (temporal, spatial) slot, spatial-invariant
        # within a temporal slot for a solid-colour clip so we can predict it
        temporal_pooled = pooled[: num_temporal * 2 : 2]  # (num_temporal, 3)
        tokens = self.proj(temporal_pooled)  # (num_temporal, hidden_dim)
        tokens = tokens.repeat_interleave(self.num_spatial, dim=0)  # (num_temporal*num_spatial, hidden)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = tokens.unsqueeze(0)  # (1, num_temporal*num_spatial, hidden)
        return out


def _mock_vjepa(seed=0, hidden_dim=1024, num_spatial=4) -> _MockVJEPA:
    torch.manual_seed(seed)
    m = _MockVJEPA(hidden_dim=hidden_dim, num_spatial=num_spatial)
    m.eval()
    return m


def test_vjepa2_name():
    enc = VJEPA2Encoder(model=_mock_vjepa())
    assert enc.name == "vjepa2-vitl"


def test_vjepa2_encode_shape_and_dtype_even_t():
    enc = VJEPA2Encoder(model=_mock_vjepa())
    clip = _fake_clip(t=6)
    z = enc.encode(clip)
    assert z.shape == (6, 1024)
    assert z.dtype == np.float32


def test_vjepa2_encode_pads_when_t_odd():
    """T=5 frames -> 2 tubelets of 2 (=4 repeated tokens), trim/pad to T=5:
    since 4 < 5, the last token is repeated once more to reach length 5."""
    enc = VJEPA2Encoder(model=_mock_vjepa())
    clip = _fake_clip(t=5)
    z = enc.encode(clip)
    assert z.shape == (5, 1024)
    np.testing.assert_array_equal(z[3], z[4])  # padded frame == last real frame


def test_vjepa2_encode_trims_when_shorter_tubelet_count():
    """T=3: floor(3/2)=1 tubelet -> repeated to 2 frames, trimmed to T=3
    would instead need padding (2 < 3); exercise the tubelet-repeat pairing
    explicitly on T=4 (exactly 2 tubelets -> 4 repeated frames, no pad/trim)."""
    enc = VJEPA2Encoder(model=_mock_vjepa())
    clip = _fake_clip(t=4)
    z = enc.encode(clip)
    assert z.shape == (4, 1024)
    # each tubelet's token is repeated twice: frame 0 == frame 1, frame 2 == frame 3
    np.testing.assert_array_equal(z[0], z[1])
    np.testing.assert_array_equal(z[2], z[3])


def test_vjepa2_cache_key_format():
    enc = VJEPA2Encoder(model=_mock_vjepa())
    assert enc.cache_key.startswith("vjepa2-vitl-")
    assert len(enc.cache_key) == len("vjepa2-vitl-") + 8


def test_vjepa2_cache_key_deterministic_same_weights():
    enc1 = VJEPA2Encoder(model=_mock_vjepa(seed=3))
    enc2 = VJEPA2Encoder(model=_mock_vjepa(seed=3))
    assert enc1.cache_key == enc2.cache_key


def test_vjepa2_cache_key_sensitive_to_weights():
    enc1 = VJEPA2Encoder(model=_mock_vjepa(seed=3))
    enc2 = VJEPA2Encoder(model=_mock_vjepa(seed=4))
    assert enc1.cache_key != enc2.cache_key


def test_vjepa2_requires_model_before_use():
    enc = VJEPA2Encoder()
    try:
        enc.encode(_fake_clip())
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    try:
        _ = enc.cache_key
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_vjepa2_encode_clip_cached_round_trip(tmp_path):
    enc = VJEPA2Encoder(model=_mock_vjepa())
    clip = _fake_clip(t=6)
    z1 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    expected = tmp_path / enc.cache_key / "deadbeef.npy"
    assert expected.exists()
    z2 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    np.testing.assert_array_equal(z1, z2)


class _MockVJEPASpatialVarying(nn.Module):
    """Unlike _MockVJEPA, spatial tokens within a temporal slot are NOT
    identical -- each spatial index adds a distinct deterministic offset --
    so mean-vs-sum-vs-select-one reductions are numerically distinguishable,
    and the expected per-frame output is hand-computable."""

    def __init__(self, hidden_dim=4, num_spatial=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_spatial = num_spatial
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, pixel_values: torch.Tensor):
        b, t, c, h, w = pixel_values.shape
        num_temporal = max(t // 2, 1)
        pooled = pixel_values.mean(dim=[3, 4])[0]  # (t, 3)
        temporal_pooled = pooled[: num_temporal * 2 : 2]  # (num_temporal, 3)
        base = self.proj(temporal_pooled)  # (num_temporal, hidden_dim)
        offsets = torch.arange(self.num_spatial, dtype=base.dtype).view(-1, 1, 1)  # (S,1,1)
        tokens = base.unsqueeze(0) + offsets  # (S, num_temporal, hidden_dim)
        tokens = tokens.transpose(0, 1).reshape(num_temporal * self.num_spatial, -1)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = tokens.unsqueeze(0)  # (1, num_temporal*num_spatial, hidden)
        return out


def test_vjepa2_spatial_mean_pipeline_correctness():
    """Pins the spatial-patch-token reduction as a MEAN (not sum/select):
    a solid-colour clip makes the per-temporal-slot base token hand-
    computable via the mock's own proj layer, and the spatial offsets
    (0, 1, ..., num_spatial-1) have a known mean, so the expected reduced
    token is exactly base + mean(offsets)."""
    hidden_dim, num_spatial = 4, 3
    torch.manual_seed(2)
    model = _MockVJEPASpatialVarying(hidden_dim=hidden_dim, num_spatial=num_spatial)
    model.eval()
    enc = VJEPA2Encoder(model=model)
    color = (30, 90, 150)
    clip = _solid_clip(t=4, color=color)  # -> 2 tubelets, no pad/trim

    z = enc.encode(clip)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalised = (np.array(color, dtype=np.float32) / 255.0 - mean) / std
    with torch.no_grad():
        base = model.proj(torch.from_numpy(normalised).unsqueeze(0)).numpy()[0]  # (hidden_dim,)
    offset_mean = np.mean(np.arange(num_spatial, dtype=np.float32))
    expected = base + offset_mean  # what mean-pooling over spatial tokens gives

    for t in range(4):
        np.testing.assert_allclose(z[t], expected, rtol=1e-4, atol=1e-5)

    # a sum-pooling regression would diverge sharply from the mean-pooled value
    wrong_sum = base + np.sum(np.arange(num_spatial, dtype=np.float32))
    assert not np.allclose(z[0], wrong_sum, rtol=1e-4, atol=1e-5)


def test_vjepa2_uses_run_backbone_hook_for_api_isolation():
    """Overriding _run_backbone alone (as a real API-mismatch fix would)
    changes encode()'s output -- confirms the reshaping/tubelet logic is
    fully downstream of that one hook."""
    enc = VJEPA2Encoder(model=_mock_vjepa())
    clip = _fake_clip(t=4)

    calls = []

    def _fake_backbone(pixel_values):
        calls.append(pixel_values.shape)
        return torch.zeros(2, 4, 1024)

    enc._run_backbone = _fake_backbone
    z = enc.encode(clip)
    assert len(calls) == 1
    np.testing.assert_array_equal(z, np.zeros((4, 1024), dtype=np.float32))
