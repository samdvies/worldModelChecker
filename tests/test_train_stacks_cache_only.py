"""TDD: scripts/train_stacks.py --cache-only flag routes
_load_pretrained_encoder through resolve_cache_only instead of the real
.load() (which downloads weights, GPU-box only). Without the flag, behaviour
is unchanged."""
import scripts.train_stacks as train_stacks


def test_load_pretrained_encoder_dinov2_default_calls_load(monkeypatch):
    calls = []

    class _FakeEncoder:
        def load(self):
            calls.append("loaded")
            return self

    monkeypatch.setattr(train_stacks, "DINOv2Encoder", lambda: _FakeEncoder())
    enc = train_stacks._load_pretrained_encoder("dinov2-s14")
    assert calls == ["loaded"]
    assert isinstance(enc, _FakeEncoder)


def test_load_pretrained_encoder_cache_only_routes_through_resolve_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        train_stacks, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )

    def _boom():
        raise AssertionError(".load() must not be called in --cache-only mode")

    monkeypatch.setattr(train_stacks, "DINOv2Encoder", lambda: type("E", (), {"load": lambda self: _boom()})())

    enc = train_stacks._load_pretrained_encoder("dinov2-s14", cache_only=True)
    assert enc is sentinel
    assert calls == ["dinov2-s14"]


def test_load_pretrained_encoder_vjepa2_cache_only(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        train_stacks, "resolve_cache_only",
        lambda name: calls.append(name) or sentinel,
    )
    enc = train_stacks._load_pretrained_encoder("vjepa2-vitl", cache_only=True)
    assert enc is sentinel
    assert calls == ["vjepa2-vitl"]


def test_cache_only_flag_routes_pretrained_stack_through_resolve_cache_only(monkeypatch, tmp_path):
    """CLI-level: --cache-only --stacks dinov2-s14 must call resolve_cache_only,
    never DINOv2Encoder().load()."""
    monkeypatch.setattr(train_stacks, "train_clips", lambda: [])
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(train_stacks, "populate_caches", lambda *a, **k: {})

    def _boom():
        raise AssertionError(".load() must not be called in --cache-only mode")

    monkeypatch.setattr(train_stacks, "DINOv2Encoder", lambda: type("E", (), {"load": lambda self: _boom()})())

    resolve_calls = []
    sentinel = type("Sentinel", (), {"cache_key": "dinov2-s14-fakehash"})()
    monkeypatch.setattr(
        train_stacks, "resolve_cache_only",
        lambda name: resolve_calls.append(name) or sentinel,
    )
    monkeypatch.setattr(
        train_stacks.sys, "argv",
        ["train_stacks.py", "--stacks", "dinov2-s14", "--cache-only"],
    )

    train_stacks.main()

    assert resolve_calls == ["dinov2-s14"]
