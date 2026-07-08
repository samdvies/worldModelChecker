"""TDD: scripts/run_mechanistic.py must resolve the two pretrained stacks
through resolve_cache_only rather than loading real weights (see
tests/test_run_report_card_encoder_loading.py -- same requirement, kept
consistent between the two scripts)."""
import scripts.run_mechanistic as run_mechanistic


def test_load_encoder_dinov2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_mechanistic, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_mechanistic._load_encoder("dinov2-s14") is sentinel
    assert calls == ["dinov2-s14"]


def test_load_encoder_vjepa2_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        run_mechanistic, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    assert run_mechanistic._load_encoder("vjepa2-vitl") is sentinel
    assert calls == ["vjepa2-vitl"]
