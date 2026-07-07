"""TDD: probes/mechanistic/data.py -- scene-disjoint probe pair iterators +
latent extraction via the frozen encoder cache."""
import numpy as np
import pytest

from physics_auditor.generator.dataset import eval_pairs, train_clips, val_clips
from physics_auditor.laws.base import MinimalPair
from physics_auditor.models.encoders import RawPixelEncoder
from physics_auditor.probes.mechanistic.data import (
    PROBE_TEST_SEEDS,
    PROBE_TRAIN_SEEDS,
    latent_frames,
    probe_pairs,
)


def test_probe_pairs_train_default_seed_range_and_count():
    pairs = probe_pairs("gravity", "train")
    assert len(pairs) == 32
    assert all(isinstance(p, MinimalPair) for p in pairs)
    assert all(p.obey.config.law == "gravity" for p in pairs)
    assert {p.obey.config.seed for p in pairs} == set(PROBE_TRAIN_SEEDS)


def test_probe_pairs_test_default_seed_range_and_count():
    pairs = probe_pairs("support", "test")
    assert len(pairs) == 16
    assert {p.obey.config.seed for p in pairs} == set(PROBE_TEST_SEEDS)


def test_probe_pairs_rejects_bad_split():
    with pytest.raises(ValueError):
        probe_pairs("gravity", "bogus")


def test_probe_seed_ranges_disjoint_from_all_existing_seed_ranges():
    probe_seeds = set(PROBE_TRAIN_SEEDS) | set(PROBE_TEST_SEEDS)
    existing = set(range(0, 16)) | set(range(100, 132)) | set(range(200, 208))
    assert probe_seeds.isdisjoint(existing)


def test_probe_train_and_test_seeds_disjoint_from_each_other():
    assert set(PROBE_TRAIN_SEEDS).isdisjoint(set(PROBE_TEST_SEEDS))


def test_probe_pairs_seeds_override_for_smoke_tests():
    pairs = probe_pairs("gravity", "train", seeds=range(300, 302))
    assert len(pairs) == 2


def test_latent_frames_returns_obey_and_violate_arrays(tmp_path):
    pairs = probe_pairs("gravity", "train", seeds=range(300, 301))
    pair = pairs[0]
    enc = RawPixelEncoder()
    obey_z, violate_z = latent_frames(pair, enc, cache_dir=str(tmp_path))
    assert obey_z.shape == (48, 256)
    assert violate_z.shape == (48, 256)
    assert not np.array_equal(obey_z, violate_z)
