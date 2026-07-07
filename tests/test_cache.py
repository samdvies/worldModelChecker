"""TDD: models/cache.py -- encode_clip_cached memoizes by encoder.cache_key/scenario_id."""
import numpy as np
import pytest

from physics_auditor.models.cache import encode_clip_cached


class _CountingEncoder:
    name = "counting"
    cache_key = "counting-v1"

    def __init__(self):
        self.calls = 0

    def encode(self, clip):
        self.calls += 1
        return np.ones((len(clip.frames), 4), dtype=np.float32) * self.calls


def _fake_clip():
    from dataclasses import dataclass

    @dataclass
    class _FakeConfig:
        scenario_id: str = "cafef00d"

    class _FakeClip:
        def __init__(self):
            self.frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            self.config = _FakeConfig()

    return _FakeClip()


def test_encode_clip_cached_writes_npy_file(tmp_path):
    enc = _CountingEncoder()
    clip = _fake_clip()
    z = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    expected = tmp_path / "counting-v1" / "cafef00d.npy"
    assert expected.exists()
    assert z.shape == (3, 4)
    assert z.dtype == np.float32


def test_encode_clip_cached_second_call_hits_cache_not_encoder(tmp_path):
    enc = _CountingEncoder()
    clip = _fake_clip()
    z1 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    z2 = encode_clip_cached(enc, clip, cache_dir=str(tmp_path))
    assert enc.calls == 1  # second call served from cache
    np.testing.assert_array_equal(z1, z2)


def test_encode_clip_cached_different_scenario_ids_dont_collide(tmp_path):
    enc = _CountingEncoder()
    clip_a = _fake_clip()
    clip_b = _fake_clip()
    clip_b.config.scenario_id = "otherid1"
    encode_clip_cached(enc, clip_a, cache_dir=str(tmp_path))
    encode_clip_cached(enc, clip_b, cache_dir=str(tmp_path))
    assert enc.calls == 2
