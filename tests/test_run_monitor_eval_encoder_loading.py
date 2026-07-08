"""TDD: scripts/run_monitor_eval.py must resolve the two pretrained stacks
(dinov2-s14, vjepa2-vitl) through resolve_cache_only rather than loading real
weights -- this script runs on machines without network/GPU access, so
DINOv2Encoder().load() / VJEPA2Encoder().load() must never be called here."""
import torch

import scripts.run_monitor_eval as run_monitor_eval
from physics_auditor.models.pretrained import CacheOnlyEncoder


def test_load_encoder_dinov2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_monitor_eval, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_monitor_eval._load_encoder("dinov2-s14") is sentinel
    assert calls == ["dinov2-s14"]


def test_load_encoder_vjepa2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_monitor_eval, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_monitor_eval._load_encoder("vjepa2-vitl") is sentinel
    assert calls == ["vjepa2-vitl"]


def test_load_predictor_for_pairs_cache_only_encoder_cache_key(monkeypatch):
    """A CacheOnlyEncoder's .cache_key must flow into load_predictor's
    encoder_cache_key= exactly like a real encoder's would."""
    cache_only_enc = CacheOnlyEncoder(name="dinov2-s14", cache_key="dinov2-s14-fakehash")

    fake_state = {"net.0.weight": torch.zeros(512, 4 * 8)}
    monkeypatch.setattr(torch, "load", lambda *a, **k: fake_state)

    captured = {}

    def fake_load_predictor(weights_path, dim, encoder_cache_key):
        captured["dim"] = dim
        captured["encoder_cache_key"] = encoder_cache_key
        return object()

    monkeypatch.setattr(run_monitor_eval, "load_predictor", fake_load_predictor)

    run_monitor_eval._load_predictor_for("dinov2-s14", cache_only_enc)

    assert captured["encoder_cache_key"] == "dinov2-s14-fakehash"
    assert captured["dim"] == 8


def test_stack_names_includes_all_five_stacks():
    assert run_monitor_eval.STACK_NAMES == [
        "raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred", "dinov2-s14", "vjepa2-vitl",
    ]


def test_predictor_weights_covers_pretrained_stacks():
    assert "dinov2-s14" in run_monitor_eval.PREDICTOR_WEIGHTS
    assert "vjepa2-vitl" in run_monitor_eval.PREDICTOR_WEIGHTS
