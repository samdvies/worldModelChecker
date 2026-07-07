"""On-disk latent cache: encode_clip_cached(encoder, clip) memoizes by
<cache_dir>/<encoder.cache_key>/<scenario_id>.npy."""
import os

import numpy as np

from physics_auditor.generator.clip import Clip
from physics_auditor.models.base import Encoder


def _cache_path(encoder: Encoder, clip: Clip, cache_dir: str) -> tuple[str, str]:
    """Returns (key_dir, path) for a given encoder/clip/cache_dir -- the
    single source of truth for the <cache_dir>/<cache_key>/<scenario_id>.npy
    scheme, shared by encode_clip_cached and encode_clips_cached."""
    key_dir = os.path.join(cache_dir, encoder.cache_key)
    path = os.path.join(key_dir, f"{clip.config.scenario_id}.npy")
    return key_dir, path


def encode_clip_cached(encoder: Encoder, clip: Clip, cache_dir: str = "cache") -> np.ndarray:
    key_dir, path = _cache_path(encoder, clip, cache_dir)

    if os.path.exists(path):
        return np.load(path)

    latents = encoder.encode(clip).astype(np.float32)
    os.makedirs(key_dir, exist_ok=True)
    np.save(path, latents)
    return latents


def encode_clips_cached(
    encoder: Encoder, clips: list[Clip], cache_dir: str = "cache", batch_size: int | None = None
) -> list[np.ndarray]:
    """Batched counterpart of encode_clip_cached: loads already-cached clips
    from disk, encodes the rest in one go (via encoder.encode_batch if
    available, else falling back to per-clip encoder.encode), writes new
    results to the same <cache_dir>/<cache_key>/<scenario_id>.npy paths, and
    returns latents in the same order as the input `clips`."""
    paths = [_cache_path(encoder, clip, cache_dir) for clip in clips]
    results: list[np.ndarray | None] = [None] * len(clips)
    uncached_idx = []

    for i, (_key_dir, path) in enumerate(paths):
        if os.path.exists(path):
            results[i] = np.load(path)
        else:
            uncached_idx.append(i)

    if uncached_idx:
        uncached_clips = [clips[i] for i in uncached_idx]
        if hasattr(encoder, "encode_batch"):
            new_latents = encoder.encode_batch(uncached_clips, batch_size=batch_size)
        else:
            new_latents = [encoder.encode(c) for c in uncached_clips]

        for idx, latents in zip(uncached_idx, new_latents):
            latents = np.asarray(latents, dtype=np.float32)
            key_dir, path = paths[idx]
            os.makedirs(key_dir, exist_ok=True)
            np.save(path, latents)
            results[idx] = latents

    return results  # type: ignore[return-value]
