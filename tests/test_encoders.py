"""TDD: RawPixelEncoder -- pure numpy/PIL downsample-to-16x16 encoder."""
import numpy as np

from physics_auditor.models.encoders import RawPixelEncoder


def _fake_clip(t=5):
    from dataclasses import dataclass

    @dataclass
    class _FakeConfig:
        scenario_id: str = "deadbeef"

    class _FakeClip:
        def __init__(self):
            self.frames = np.random.default_rng(0).integers(0, 256, size=(t, 64, 64, 3), dtype=np.uint8)
            self.config = _FakeConfig()

    return _FakeClip()


def test_raw_pixel_encoder_name_and_cache_key():
    enc = RawPixelEncoder()
    assert enc.name == "raw-pixel"
    assert enc.cache_key == "raw16-v1"


def test_raw_pixel_encoder_output_shape_and_dtype():
    enc = RawPixelEncoder()
    clip = _fake_clip(t=7)
    z = enc.encode(clip)
    assert z.shape == (7, 256)
    assert z.dtype == np.float32


def test_raw_pixel_encoder_values_in_unit_range():
    enc = RawPixelEncoder()
    clip = _fake_clip(t=3)
    z = enc.encode(clip)
    assert z.min() >= 0.0
    assert z.max() <= 1.0
