"""TDD: TinyCNNAE + TinyCNNPred torch encoder stacks.

Fast synthetic-tensor tests only (2 epochs, 32 random frames) -- real
training on the full lawful dataset is run separately via
scripts/train_stacks.py, not in the test suite.
"""
import numpy as np
import torch

from physics_auditor.models.encoders import TinyCNNAE, TinyCNNPred


def _fake_clip(t=5, seed=0):
    from dataclasses import dataclass

    @dataclass
    class _FakeConfig:
        scenario_id: str = "deadbeef"

    class _FakeClip:
        def __init__(self):
            self.frames = np.random.default_rng(seed).integers(0, 256, size=(t, 64, 64, 3), dtype=np.uint8)
            self.config = _FakeConfig()

    return _FakeClip()


def _random_frame_batch(n=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8).astype(np.float32) / 255.0)


# ---------------------------------------------------------------- TinyCNNAE

def test_tiny_cnn_ae_name():
    enc = TinyCNNAE()
    assert enc.name == "tiny-cnn-ae"


def test_tiny_cnn_ae_cache_key_format():
    enc = TinyCNNAE()
    assert enc.cache_key.startswith("tiny-cnn-ae-")
    assert len(enc.cache_key) == len("tiny-cnn-ae-") + 8


def test_tiny_cnn_ae_encode_shape_and_dtype():
    enc = TinyCNNAE()
    clip = _fake_clip(t=6)
    z = enc.encode(clip)
    assert z.shape == (6, 128)
    assert z.dtype == np.float32


def test_tiny_cnn_ae_encode_is_deterministic_eval_mode():
    enc = TinyCNNAE()
    clip = _fake_clip(t=4)
    z1 = enc.encode(clip)
    z2 = enc.encode(clip)
    np.testing.assert_array_equal(z1, z2)


def test_tiny_cnn_ae_cache_key_changes_after_training():
    torch.manual_seed(0)
    enc = TinyCNNAE()
    key_before = enc.cache_key
    frames = _random_frame_batch(n=32, seed=1)
    enc.fit(frames, epochs=2, batch_size=8, seed=0)
    key_after = enc.cache_key
    assert key_before != key_after


def test_tiny_cnn_ae_fit_reduces_loss():
    torch.manual_seed(0)
    enc = TinyCNNAE()
    frames = _random_frame_batch(n=32, seed=2)
    losses = enc.fit(frames, epochs=2, batch_size=8, seed=0)
    assert len(losses) == 2
    assert all(np.isfinite(losses))


def test_tiny_cnn_ae_fit_deterministic_same_seed():
    frames = _random_frame_batch(n=32, seed=3)

    enc1 = TinyCNNAE()
    losses1 = enc1.fit(frames, epochs=2, batch_size=8, seed=0)

    enc2 = TinyCNNAE()
    losses2 = enc2.fit(frames, epochs=2, batch_size=8, seed=0)

    np.testing.assert_allclose(losses1, losses2)
    clip = _fake_clip(t=3)
    np.testing.assert_array_equal(enc1.encode(clip), enc2.encode(clip))


# -------------------------------------------------------------- TinyCNNPred

def test_tiny_cnn_pred_name():
    enc = TinyCNNPred()
    assert enc.name == "tiny-cnn-pred"


def test_tiny_cnn_pred_cache_key_format():
    enc = TinyCNNPred()
    assert enc.cache_key.startswith("tiny-cnn-pred-")
    assert len(enc.cache_key) == len("tiny-cnn-pred-") + 8


def test_tiny_cnn_pred_encode_shape_and_dtype():
    enc = TinyCNNPred()
    clip = _fake_clip(t=6)
    z = enc.encode(clip)
    assert z.shape == (6, 128)
    assert z.dtype == np.float32


def test_tiny_cnn_pred_cache_key_changes_after_training():
    torch.manual_seed(0)
    enc = TinyCNNPred()
    key_before = enc.cache_key
    frames = _random_frame_batch(n=40, seed=4)  # >= 5 for one 5-frame window
    enc.fit(frames, epochs=2, batch_size=8, seed=0)
    key_after = enc.cache_key
    assert key_before != key_after


def test_tiny_cnn_pred_fit_reduces_loss_len():
    torch.manual_seed(0)
    enc = TinyCNNPred()
    frames = _random_frame_batch(n=40, seed=5)
    losses = enc.fit(frames, epochs=2, batch_size=8, seed=0)
    assert len(losses) == 2
    assert all(np.isfinite(losses))


def test_tiny_cnn_pred_fit_deterministic_same_seed():
    frames = _random_frame_batch(n=40, seed=6)

    enc1 = TinyCNNPred()
    losses1 = enc1.fit(frames, epochs=2, batch_size=8, seed=0)

    enc2 = TinyCNNPred()
    losses2 = enc2.fit(frames, epochs=2, batch_size=8, seed=0)

    np.testing.assert_allclose(losses1, losses2)
    clip = _fake_clip(t=3)
    np.testing.assert_array_equal(enc1.encode(clip), enc2.encode(clip))


def test_tiny_cnn_ae_and_pred_cache_keys_differ():
    ae = TinyCNNAE()
    pred = TinyCNNPred()
    assert ae.cache_key != pred.cache_key
