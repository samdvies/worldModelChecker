"""TDD: generator/dataset.py -- disjoint train/val/eval clip and pair iterators."""
from physics_auditor.generator.clip import Clip
from physics_auditor.generator.dataset import eval_pairs, train_clips, val_clips
from physics_auditor.laws.base import MinimalPair


def test_train_clips_returns_obey_clips_across_all_four_laws():
    clips = list(train_clips(seeds=range(100, 102)))
    # 2 seeds x 4 laws = 8 clips
    assert len(clips) == 8
    assert all(isinstance(c, Clip) for c in clips)
    assert all(not c.config.violate for c in clips)


def test_val_clips_returns_obey_clips_across_all_four_laws():
    clips = list(val_clips(seeds=range(200, 201)))
    assert len(clips) == 4
    assert all(not c.config.violate for c in clips)


def test_eval_pairs_returns_minimal_pairs_for_named_law():
    pairs = eval_pairs("gravity", seeds=range(0, 3))
    assert len(pairs) == 3
    assert all(isinstance(p, MinimalPair) for p in pairs)
    assert all(p.obey.config.law == "gravity" for p in pairs)


def test_train_and_eval_seed_ranges_are_scene_disjoint():
    train = {c.config.seed for c in train_clips(seeds=range(100, 132))}
    eval_seeds = set(range(0, 64))
    assert train.isdisjoint(eval_seeds)


def test_eval_pairs_default_seed_range_is_64():
    pairs = eval_pairs("gravity")
    assert len(pairs) == 64
    assert {p.obey.config.seed for p in pairs} == set(range(0, 64))
