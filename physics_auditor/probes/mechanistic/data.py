"""Scene-disjoint pair iterators + per-frame latent extraction for the
mechanistic (decodability / intervention) probes.

Seed ranges are SACRED and disjoint from every other seed range used in this
repo: probe-train pairs use seeds 300..331 (32 pairs/law), probe-test pairs
use seeds 400..415 (16 pairs/law). Probes trained on the 300s may NEVER be
evaluated in a way that lets them see the 400s during fitting.
"""
from collections.abc import Iterable

import numpy as np

from physics_auditor.laws import ALL_LAWS
from physics_auditor.laws.base import MinimalPair
from physics_auditor.models.base import Encoder
from physics_auditor.models.cache import encode_clip_cached

PROBE_TRAIN_SEEDS: range = range(300, 332)  # 32 seeds
PROBE_TEST_SEEDS: range = range(400, 416)  # 16 seeds

_SPLIT_SEEDS: dict[str, range] = {
    "train": PROBE_TRAIN_SEEDS,
    "test": PROBE_TEST_SEEDS,
}


def probe_pairs(law_name: str, split: str, seeds: Iterable[int] | None = None) -> list[MinimalPair]:
    """MinimalPairs for one law's mechanistic-probe split ('train' or 'test').

    Seeds default to the sacred probe ranges; an explicit `seeds` override is
    accepted only for testing convenience (e.g. a 2-seed smoke test) -- real
    probe code must call this with the default.
    """
    if split not in _SPLIT_SEEDS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    law = ALL_LAWS[law_name]()
    use_seeds = seeds if seeds is not None else _SPLIT_SEEDS[split]
    return [law.generate_pair(seed) for seed in use_seeds]


def latent_frames(pair: MinimalPair, stack_encoder: Encoder, cache_dir: str = "cache") -> tuple[np.ndarray, np.ndarray]:
    """Per-frame latents (T, D) for a pair's obey and violate clips, via the
    frozen on-disk cache (same encoder + cache semantics as everywhere else)."""
    obey_z = encode_clip_cached(stack_encoder, pair.obey, cache_dir=cache_dir)
    violate_z = encode_clip_cached(stack_encoder, pair.violate, cache_dir=cache_dir)
    return obey_z, violate_z
