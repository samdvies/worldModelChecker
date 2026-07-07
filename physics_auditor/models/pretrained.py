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

Throughput (T4 GPU box only -- see docs/failure-sweeps.md class G and
scripts/aws/runbook.md): on cuda, `.load()` casts the model to fp16 and
`encode`/`encode_batch` cast inputs to match; the returned latents are
always cast back to float32 so cached .npy files and downstream predictor
code are unaffected. V-JEPA-2 additionally batches MULTIPLE CLIPS per
forward via `encode_batch` (pixel_values_videos supports a real batch
dimension). Every cuda-only branch is gated behind
`torch.cuda.is_available()`; the actual half/batched math is implemented in
device-agnostic `_encode_core`/`_encode_core_batch` methods that take
`device`/`use_half` as explicit arguments, so tests exercise the exact same
code path on CPU hardware (device="cpu", use_half=True) as a CPU-equivalent
proof of the cuda behaviour, without ever needing a real GPU.
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


def _device_plan(default_cpu_batch: int, default_cuda_batch: int) -> tuple[str, bool, int]:
    """The ONLY place that consults torch.cuda.is_available() for this
    module: returns (device, use_half, default_batch_size). Trivial by
    design -- all the actual behaviour it selects between is implemented in
    device-agnostic core methods that tests call directly with explicit
    device/use_half, so this branch itself needs no CPU-equivalent test."""
    if torch.cuda.is_available():
        return "cuda", True, default_cuda_batch
    return "cpu", False, default_cpu_batch


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
        machine with network access to torch.hub / GitHub. On a cuda box,
        also casts the model to fp16 for throughput (see module docstring)."""
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.half().to("cuda")
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

    def _encode_core(self, clip: Clip, device: str, use_half: bool) -> np.ndarray:
        """Device-agnostic encode core: identical math regardless of device,
        so tests can exercise the cuda-shaped path (use_half=True) entirely
        on CPU hardware by passing device='cpu', use_half=True."""
        model = self._require_model()
        model.eval()
        x = _resize_and_normalize(clip.frames, self.INPUT_SIZE)  # (T,3,224,224)
        model.to(device)
        x = x.to(device)
        if use_half:
            model.half()
            x = x.half()
        with torch.no_grad():
            z = model(x)  # CLS embedding, (T, 384)
        return z.detach().float().cpu().numpy().astype(np.float32)

    def encode(self, clip: Clip) -> np.ndarray:
        device, use_half, _ = _device_plan(default_cpu_batch=1, default_cuda_batch=1)
        return self._encode_core(clip, device=device, use_half=use_half)


class VJEPA2Encoder:
    """V-JEPA-2 ViT-L adapter (facebook/vjepa2-vitl-fpc64-256, HuggingFace
    transformers -- 'gpu' optional dependency group). V-JEPA-2 tokenises
    time in tubelets of 2 frames; per-frame latents are produced by
    repeating each temporal token twice, then trimming/padding to T. The
    latent for each temporal slot is the mean over its spatial patch
    tokens.

    `encode_batch` packs multiple clips into a single forward's batch
    dimension (pixel_values_videos: (B, T, 3, 256, 256)) for GPU throughput;
    it requires every clip in a given internal chunk to share the same
    frame count T (true for this dataset's train/val/eval clip sets)."""

    name = "vjepa2-vitl"
    INPUT_SIZE = 256
    TUBELET_SIZE = 2
    HF_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
    DEFAULT_CPU_BATCH_SIZE = 1
    DEFAULT_CUDA_BATCH_SIZE = 4

    def __init__(self, model: nn.Module | None = None):
        self.model = model

    def load(self) -> "VJEPA2Encoder":
        """Lazily downloads real V-JEPA-2 weights via transformers. Only
        safe to call on a machine with network access to the HF hub, and
        with the 'gpu' optional dependency group installed. On a cuda box,
        also casts the model to fp16 for throughput (see module docstring)."""
        from transformers import AutoModel  # gpu-group-only import

        self.model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.half().to("cuda")
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
        hidden_dim). Isolated on purpose (see module docstring). Unbatched
        (B=1) single-clip path -- used by `encode`."""
        with torch.no_grad():
            out = self.model(pixel_values_videos=pixel_values)
        hidden = out.last_hidden_state[0]  # (num_temporal*num_spatial, hidden_dim)
        num_temporal = max(pixel_values.shape[1] // self.TUBELET_SIZE, 1)
        num_spatial = hidden.shape[0] // num_temporal
        assert hidden.shape[0] % num_temporal == 0, (
            f"hidden tokens {hidden.shape[0]} not evenly divisible by "
            f"num_temporal {num_temporal}"
        )
        return hidden[: num_temporal * num_spatial].view(num_temporal, num_spatial, -1)

    def _run_backbone_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Batched counterpart of `_run_backbone`: pixel_values (B, T, 3,
        256, 256) -> (B, num_temporal, num_spatial, hidden_dim)."""
        with torch.no_grad():
            out = self.model(pixel_values_videos=pixel_values)
        hidden = out.last_hidden_state  # (B, num_temporal*num_spatial, hidden_dim)
        b, n_tokens, hidden_dim = hidden.shape
        num_temporal = max(pixel_values.shape[1] // self.TUBELET_SIZE, 1)
        num_spatial = n_tokens // num_temporal
        assert n_tokens % num_temporal == 0, (
            f"hidden tokens {n_tokens} not evenly divisible by num_temporal {num_temporal}"
        )
        return hidden[:, : num_temporal * num_spatial].reshape(b, num_temporal, num_spatial, hidden_dim)

    def _postprocess_per_temporal(self, per_temporal: torch.Tensor, t: int) -> torch.Tensor:
        """per_temporal: (num_temporal, hidden) -> (t, hidden): repeat each
        temporal token TUBELET_SIZE times, then trim/pad to exactly t."""
        per_frame = per_temporal.repeat_interleave(self.TUBELET_SIZE, dim=0)
        if per_frame.shape[0] >= t:
            per_frame = per_frame[:t]
        else:
            pad_n = t - per_frame.shape[0]
            pad = per_frame[-1:].repeat(pad_n, 1)
            per_frame = torch.cat([per_frame, pad], dim=0)
        return per_frame

    def _encode_core(self, clip: Clip, device: str, use_half: bool) -> np.ndarray:
        """Device-agnostic single-clip encode core (see DINOv2Encoder for
        why this split exists)."""
        model = self._require_model()
        model.eval()
        t = clip.frames.shape[0]
        x = _resize_and_normalize(clip.frames, self.INPUT_SIZE).unsqueeze(0)  # (1,T,3,256,256)
        model.to(device)
        x = x.to(device)
        if use_half:
            model.half()
            x = x.half()

        backbone_out = self._run_backbone(x)  # (num_temporal, num_spatial, hidden_dim)
        per_temporal = backbone_out.mean(dim=1)  # (num_temporal, hidden_dim)
        per_frame = self._postprocess_per_temporal(per_temporal, t)
        return per_frame.detach().float().cpu().numpy().astype(np.float32)

    def encode(self, clip: Clip) -> np.ndarray:
        device, use_half, _ = _device_plan(
            default_cpu_batch=self.DEFAULT_CPU_BATCH_SIZE,
            default_cuda_batch=self.DEFAULT_CUDA_BATCH_SIZE,
        )
        return self._encode_core(clip, device=device, use_half=use_half)

    def _encode_core_batch(
        self, clips: list[Clip], device: str, use_half: bool, batch_size: int
    ) -> list[np.ndarray]:
        """Device-agnostic multi-clip batched encode core. Clips are
        chunked into groups of at most `batch_size`; every clip within one
        chunk must share the same frame count T (chunks are formed
        contiguously from the caller-supplied order, so callers should
        group same-length clips together for real batching benefit -- a
        clip whose T differs from its chunk-mates falls back to its own
        singleton chunk rather than erroring)."""
        model = self._require_model()
        model.eval()
        model.to(device)

        outputs: list[np.ndarray] = [None] * len(clips)  # type: ignore[list-item]
        i = 0
        n = len(clips)
        while i < n:
            t0 = clips[i].frames.shape[0]
            j = i + 1
            while j < min(i + batch_size, n) and clips[j].frames.shape[0] == t0:
                j += 1
            chunk = clips[i:j]

            xs = [_resize_and_normalize(c.frames, self.INPUT_SIZE) for c in chunk]  # each (T,3,256,256)
            x = torch.stack(xs, dim=0).to(device)  # (B,T,3,256,256)
            if use_half:
                model.half()
                x = x.half()

            backbone_out = self._run_backbone_batch(x)  # (B, num_temporal, num_spatial, hidden)
            per_temporal = backbone_out.mean(dim=2)  # (B, num_temporal, hidden)

            for offset, c in enumerate(chunk):
                per_frame = self._postprocess_per_temporal(per_temporal[offset], c.frames.shape[0])
                outputs[i + offset] = per_frame.detach().float().cpu().numpy().astype(np.float32)

            i = j
        return outputs

    def encode_batch(self, clips: list[Clip], batch_size: int | None = None) -> list[np.ndarray]:
        """Encode multiple clips, batching `batch_size` clips per forward
        (default 4 on cuda, 1 on cpu -- see module docstring). Returns a
        list of per-clip (T, 1024) float32 latents, identical (within fp
        tolerance) to calling `encode` on each clip individually."""
        device, use_half, default_batch = _device_plan(
            default_cpu_batch=self.DEFAULT_CPU_BATCH_SIZE,
            default_cuda_batch=self.DEFAULT_CUDA_BATCH_SIZE,
        )
        if batch_size is None:
            batch_size = default_batch
        return self._encode_core_batch(clips, device=device, use_half=use_half, batch_size=batch_size)
