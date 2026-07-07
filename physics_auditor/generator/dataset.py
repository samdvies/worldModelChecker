"""Scene-disjoint dataset iterators for training/validating/evaluating on
LAWFUL (obey) clips and eval MinimalPairs.

Seed ranges are SACRED: eval pair seeds range(0,16); predictor/encoder train
seeds range(100,132); val seeds range(200,208). Nothing from eval seeds may
ever be seen in training.
"""
from collections.abc import Iterable, Iterator

from physics_auditor.generator.clip import Clip
from physics_auditor.laws import ALL_LAWS
from physics_auditor.laws.base import MinimalPair


def _obey_clips(seeds: Iterable[int]) -> Iterator[Clip]:
    for seed in seeds:
        for law_cls in ALL_LAWS.values():
            law = law_cls()
            yield law.generate_pair(seed).obey


def train_clips(seeds: Iterable[int] = range(100, 132)) -> Iterator[Clip]:
    """LAWFUL (obey) clips across all four laws, for training."""
    return _obey_clips(seeds)


def val_clips(seeds: Iterable[int] = range(200, 208)) -> Iterator[Clip]:
    """LAWFUL (obey) clips across all four laws, for validation."""
    return _obey_clips(seeds)


def eval_pairs(law_name: str, seeds: Iterable[int] = range(0, 16)) -> list[MinimalPair]:
    """MinimalPairs (obey + violate) for a single named law, for evaluation."""
    law = ALL_LAWS[law_name]()
    return [law.generate_pair(seed) for seed in seeds]
