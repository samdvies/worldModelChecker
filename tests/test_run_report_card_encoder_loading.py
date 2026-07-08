"""TDD: scripts/run_report_card.py must resolve the two pretrained stacks
(dinov2-s14, vjepa2-vitl) through resolve_cache_only rather than loading real
weights -- these scripts run on machines without network/GPU access, so
DINOv2Encoder().load() / VJEPA2Encoder().load() must never be called here."""
import torch

import scripts.run_report_card as run_report_card
from physics_auditor.models.pretrained import CacheOnlyEncoder


def test_load_encoder_dinov2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_report_card, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_report_card._load_encoder("dinov2-s14") is sentinel
    assert calls == ["dinov2-s14"]


def test_load_encoder_vjepa2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_report_card, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_report_card._load_encoder("vjepa2-vitl") is sentinel
    assert calls == ["vjepa2-vitl"]


def test_build_adapters_pairs_cache_only_encoder_cache_key_through_load_predictor(monkeypatch):
    """A CacheOnlyEncoder's .cache_key must flow into load_predictor's
    encoder_cache_key= exactly like a real encoder's would -- the pairing
    assertion in predictor.py doesn't care whether the encoder can actually
    encode(), only that cache_key matches what the predictor was trained
    against."""
    cache_only_enc = CacheOnlyEncoder(name="dinov2-s14", cache_key="dinov2-s14-fakehash")
    monkeypatch.setattr(run_report_card, "_load_encoder", lambda name: cache_only_enc)

    fake_state = {"net.0.weight": torch.zeros(512, 4 * 8)}
    monkeypatch.setattr(torch, "load", lambda *a, **k: fake_state)

    captured = {}

    def fake_load_predictor(weights_path, dim, encoder_cache_key):
        captured["dim"] = dim
        captured["encoder_cache_key"] = encoder_cache_key
        return object()

    monkeypatch.setattr(run_report_card, "load_predictor", fake_load_predictor)
    monkeypatch.setattr(run_report_card, "load_predictor_meta", lambda path: {})

    run_report_card.build_adapters(stacks=["dinov2-s14"])

    assert captured["encoder_cache_key"] == "dinov2-s14-fakehash"
    assert captured["dim"] == 8
