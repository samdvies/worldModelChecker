"""TDD: models/cache.py -- encode_clip_cached memoizes by encoder.cache_key/scenario_id."""
import os

import numpy as np
import pytest

from physics_auditor.models.cache import encode_clip_cached, encode_clips_cached


class _CountingEncoder:
    name = "counting"
    cache_key = "counting-v1"

    def __init__(self):
        self.calls = 0

    def encode(self, clip):
        self.calls += 1
        return np.ones((len(clip.frames), 4), dtype=np.float32) * self.calls


class _BatchEncoder:
    """Fake encoder WITH encode_batch, distinguishable output pattern (9s)
    from the per-clip fallback pattern (1s, 2s, ... via _CountingEncoder-style
    encode) so tests can assert the batch path was actually used."""

    name = "batching"
    cache_key = "batching-v1"

    def __init__(self):
        self.encode_calls = 0
        self.encode_batch_calls = 0

    def encode(self, clip):
        self.encode_calls += 1
        return np.ones((len(clip.frames), 4), dtype=np.float32)

    def encode_batch(self, clips, batch_size=None):
        self.encode_batch_calls += 1
        return [np.full((len(c.frames), 4), 9.0, dtype=np.float32) for c in clips]


def _fake_clip(scenario_id="cafef00d"):
    from dataclasses import dataclass

    @dataclass
    class _FakeConfig:
        scenario_id: str = "cafef00d"

    class _FakeClip:
        def __init__(self):
            self.frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            self.config = _FakeConfig(scenario_id=scenario_id)

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


def test_encode_clips_cached_mixed_preserves_input_order(tmp_path):
    enc = _CountingEncoder()
    clip_a = _fake_clip("aaaaaaaa")
    clip_b = _fake_clip("bbbbbbbb")
    clip_c = _fake_clip("cccccccc")

    # Pre-warm the cache for clip_b only.
    z_b_precomputed = encode_clip_cached(enc, clip_b, cache_dir=str(tmp_path))

    results = encode_clips_cached(enc, [clip_a, clip_b, clip_c], cache_dir=str(tmp_path))

    assert len(results) == 3
    np.testing.assert_array_equal(results[1], z_b_precomputed)
    # clip_a and clip_c were freshly encoded (order-position, not value, is
    # what's under test -- values just need to correspond to their clip).
    expected_a = np.load(os.path.join(str(tmp_path), enc.cache_key, "aaaaaaaa.npy"))
    expected_c = np.load(os.path.join(str(tmp_path), enc.cache_key, "cccccccc.npy"))
    np.testing.assert_array_equal(results[0], expected_a)
    np.testing.assert_array_equal(results[2], expected_c)


def test_encode_clips_cached_falls_back_to_encode_per_clip(tmp_path):
    enc = _CountingEncoder()  # no encode_batch attribute
    clips = [_fake_clip("id000001"), _fake_clip("id000002"), _fake_clip("id000003")]

    results = encode_clips_cached(enc, clips, cache_dir=str(tmp_path))

    assert enc.calls == 3
    assert len(results) == 3


def test_encode_clips_cached_uses_encode_batch_when_available(tmp_path):
    enc = _BatchEncoder()
    clips = [_fake_clip("id100001"), _fake_clip("id100002"), _fake_clip("id100003")]

    results = encode_clips_cached(enc, clips, cache_dir=str(tmp_path))

    assert enc.encode_batch_calls == 1
    assert enc.encode_calls == 0
    for z in results:
        np.testing.assert_array_equal(z, np.full((3, 4), 9.0, dtype=np.float32))


def test_encode_clips_cached_writes_to_same_paths_as_encode_clip_cached(tmp_path):
    enc = _CountingEncoder()
    clip = _fake_clip("deadbeef")

    encode_clips_cached(enc, [clip], cache_dir=str(tmp_path))
    expected_path = os.path.join(str(tmp_path), enc.cache_key, "deadbeef.npy")
    assert os.path.exists(expected_path)

    # A second, independent encoder instance hitting the same cache dir via
    # encode_clip_cached must see the file encode_clips_cached wrote (same
    # path scheme) and must NOT re-encode.
    enc2 = _CountingEncoder()
    z = encode_clip_cached(enc2, clip, cache_dir=str(tmp_path))
    assert enc2.calls == 0
    np.testing.assert_array_equal(z, np.ones((3, 4), dtype=np.float32))
