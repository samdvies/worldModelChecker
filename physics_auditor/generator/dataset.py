"""Scene-disjoint dataset iterators for training/validating/evaluating on
LAWFUL (obey) clips and eval MinimalPairs.

Seed ranges are SACRED: eval pair seeds range(0,64); predictor/encoder train
seeds range(100,132); val seeds range(200,208). Also disjoint (owned by the
mechanistic probes, not this module): probe-train seeds range(300,332),
probe-test seeds range(400,416). Nothing from eval seeds may ever be seen in
training.
"""
from collections.abc import Iterable, Iterator

from physics_auditor.generator.clip import Clip
from physics_auditor.laws import ALL_LAWS, TRAIN_LAWS
from physics_auditor.laws.base import MinimalPair


def _obey_clips(seeds: Iterable[int]) -> Iterator[Clip]:
    for seed in seeds:
        for law_cls in TRAIN_LAWS.values():
            law = law_cls()
            yield law.generate_pair(seed).obey


def train_clips(seeds: Iterable[int] = range(100, 132)) -> Iterator[Clip]:
    """LAWFUL (obey) clips across the four TRAIN_LAWS only, for training.

    EVAL-ONLY laws (permanence-ext, support-hard) are deliberately excluded so
    that existing trained predictor/encoder weights and caches stay valid.
    """
    return _obey_clips(seeds)


def val_clips(seeds: Iterable[int] = range(200, 208)) -> Iterator[Clip]:
    """LAWFUL (obey) clips across the four TRAIN_LAWS only, for validation.

    EVAL-ONLY laws (permanence-ext, support-hard) are deliberately excluded so
    that existing trained predictor/encoder weights and caches stay valid.
    """
    return _obey_clips(seeds)


def eval_pairs(law_name: str, seeds: Iterable[int] = range(0, 64)) -> list[MinimalPair]:
    """MinimalPairs (obey + violate) for a single named law, for evaluation."""
    law = ALL_LAWS[law_name]()
    return [law.generate_pair(seed) for seed in seeds]
