"""Pretrained encoder adapters (DINOv2, V-JEPA-2). Both satisfy the same
Encoder protocol as the from-scratch stacks in encoders.py (name + cache_key
+ encode(clip) -> (T, D) float32), so they drop into LatentStackAdapter and
encode_clip_cached unchanged.

CRITICAL: no weights may be downloaded on a machine without network access
to the model hubs. Construction never loads weights -- pass an already
loaded `model=` (tests inject a mock nn.Module), or call `.load()` on a
GPU box with network access, which is the ONLY place a hub/HF download may
happen. transformers (needed by VJEPA2Encoder.load) is an optional 'gpu'
dependency and is imported lazily inside `.load()` only, never at module
import time, so the core install stays clean.
"""
import hashlib
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from physics_auditor.generator.clip import Clip

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _state_dict_hash(module: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return hashlib.sha1(buf.getvalue()).hexdigest()[:8]


def _resize_and_normalize(frames: np.ndarray, size: int) -> torch.Tensor:
    """frames: (T, 64, 64, 3) uint8 -> (T, 3, size, size) float32 tensor,
    resized with bicubic interpolation and ImageNet-normalised."""
    t = frames.shape[0]
    out = np.empty((t, size, size, 3), dtype=np.float32)
    for i in range(t):
        img = Image.fromarray(frames[i]).resize((size, size), Image.BICUBIC)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    out = (out - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().float()  # (T,3,size,size)


class DINOv2Encoder:
    """DINOv2 ViT-S/14 adapter. Construction takes an optional pre-loaded
    `model` (injected in tests); `.load()` fetches real pretrained weights
    via torch.hub -- GPU-box only, never called by tests."""

    name = "dinov2-s14"
    INPUT_SIZE = 224

    def __init__(self, model: nn.Module | None = None):
        self.model = model

    def load(self) -> "DINOv2Encoder":
        """Lazily downloads real DINOv2 weights. Only safe to call on a
        machine with network access to torch.hub / GitHub."""
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        return self

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError(
                f"{self.name}: no model loaded -- call .load() (GPU box) or "
                f"construct with model=<mock> (tests) first"
            )
        return self.model

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{_state_dict_hash(self._require_model())}"

    def encode(self, clip: Clip) -> np.ndarray:
        model = self._require_model()
        model.eval()
        x = _resize_and_normalize(clip.frames, self.INPUT_SIZE)  # (T,3,224,224)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        x = x.to(device)
        with torch.no_grad():
            z = model(x)  # CLS embedding, (T, 384)
        return z.detach().cpu().numpy().astype(np.float32)


class VJEPA2Encoder:
    """V-JEPA-2 ViT-L adapter (facebook/vjepa2-vitl-fpc64-256, HuggingFace
    transformers -- 'gpu' optional dependency group). V-JEPA-2 tokenises
    time in tubelets of 2 frames; per-frame latents are produced by
    repeating each temporal token twice, then trimming/padding to T. The
    latent for each temporal slot is the mean over its spatial patch
    tokens."""

    name = "vjepa2-vitl"
    INPUT_SIZE = 256
    TUBELET_SIZE = 2
    HF_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

    def __init__(self, model: nn.Module | None = None):
        self.model = model

    def load(self) -> "VJEPA2Encoder":
        """Lazily downloads real V-JEPA-2 weights via transformers. Only
        safe to call on a machine with network access to the HF hub, and
        with the 'gpu' optional dependency group installed."""
        from transformers import AutoModel  # gpu-group-only import

        self.model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self.model.eval()
        return self

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError(
                f"{self.name}: no model loaded -- call .load() (GPU box) or "
                f"construct with model=<mock> (tests) first"
            )
        return self.model

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{_state_dict_hash(self._require_model())}"

    def _run_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """THE one function to fix if the HF V-JEPA-2 API changes shape.
        pixel_values: (1, T, 3, 256, 256) -> (num_temporal, num_spatial,
        hidden_dim). Isolated on purpose (see module docstring)."""
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        hidden = out.last_hidden_state[0]  # (num_temporal*num_spatial, hidden_dim)
        num_temporal = max(pixel_values.shape[1] // self.TUBELET_SIZE, 1)
        num_spatial = hidden.shape[0] // num_temporal
        return hidden[: num_temporal * num_spatial].view(num_temporal, num_spatial, -1)

    def encode(self, clip: Clip) -> np.ndarray:
        model = self._require_model()
        model.eval()
        t = clip.frames.shape[0]
        x = _resize_and_normalize(clip.frames, self.INPUT_SIZE).unsqueeze(0)  # (1,T,3,256,256)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        x = x.to(device)

        backbone_out = self._run_backbone(x)  # (num_temporal, num_spatial, hidden_dim)
        per_temporal = backbone_out.mean(dim=1)  # (num_temporal, hidden_dim)
        per_frame = per_temporal.repeat_interleave(self.TUBELET_SIZE, dim=0)  # (2*num_temporal, hidden)

        if per_frame.shape[0] >= t:
            per_frame = per_frame[:t]
        else:
            pad_n = t - per_frame.shape[0]
            pad = per_frame[-1:].repeat(pad_n, 1)
            per_frame = torch.cat([per_frame, pad], dim=0)

        return per_frame.detach().cpu().numpy().astype(np.float32)
