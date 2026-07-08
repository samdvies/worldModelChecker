"""TDD: scripts/train_stacks.py's cache-warm stack-selection plan.

Regression guard for docs/failure-sweeps.md class G: a GPU run invoked as
`--stacks dinov2-s14,vjepa2-vitl` wasted ~2h re-encoding the three LOCAL
stacks (raw-pixel, tiny-cnn-ae, tiny-cnn-pred) anyway. `plan_stacks` is the
pure, side-effect-free function the cache-warm step consults to decide
which stacks to build/encode -- tested here directly (no real run, no
network, no torch model construction) so this can never regress silently.
"""
import pytest

from scripts.train_stacks import ALL_STACK_NAMES, DEFAULT_STACKS, PRETRAINED_STACKS, plan_stacks


def test_plan_stacks_pretrained_only_excludes_all_local_stacks():
    """The exact shape of the incident: --stacks dinov2-s14,vjepa2-vitl must
    plan ZERO local-stack work."""
    plan = plan_stacks(["dinov2-s14", "vjepa2-vitl"])
    local_work = [name for name in plan if name in DEFAULT_STACKS]
    assert local_work == [], (
        f"requesting only pretrained stacks must not plan any local-stack "
        f"work, got {local_work} (class G regression -- see docs/failure-sweeps.md)"
    )
    assert plan == ["dinov2-s14", "vjepa2-vitl"]


def test_plan_stacks_single_pretrained_stack():
    assert plan_stacks(["dinov2-s14"]) == ["dinov2-s14"]


def test_plan_stacks_default_local_stacks_unchanged():
    """Legacy behaviour: no --stacks flag -> DEFAULT_STACKS, in order."""
    assert plan_stacks(DEFAULT_STACKS) == DEFAULT_STACKS


def test_plan_stacks_mixed_local_and_pretrained():
    plan = plan_stacks(["vjepa2-vitl", "tiny-cnn-ae"])
    assert plan == ["tiny-cnn-ae", "vjepa2-vitl"]  # build order: local stacks first, then pretrained


def test_plan_stacks_preserves_requested_subset_only():
    plan = plan_stacks(["raw-pixel"])
    assert plan == ["raw-pixel"]
    assert "tiny-cnn-ae" not in plan
    assert "tiny-cnn-pred" not in plan


@pytest.mark.parametrize("name", ALL_STACK_NAMES)
def test_plan_stacks_every_known_stack_name_plannable_alone(name):
    assert plan_stacks([name]) == [name]


class _FakePair:
    def __init__(self, obey, violate):
        self.obey = obey
        self.violate = violate


def test_probe_clips_fan_out_matches_4_laws_x_2_splits_x_pairs_x_2(monkeypatch):
    """docs/failure-sweeps.md class I: remote_job.sh used to warm the
    mechanistic-probe latent caches via run_report_card.py/run_mechanistic.py
    on the GPU box, both of which crash there (predictor weights never
    trained on the box) -- probe_clips() is the pure fan-out those scripts'
    cache-warming incidentally did, now the single source of truth for it.
    Real probe_pairs() is real pymunk generation (seeds 300..331/400..415,
    576 clips across all 6 ALL_LAWS) -- monkeypatched here with a tiny fake
    to stay fast."""
    import scripts.train_stacks as train_stacks

    calls = []

    def fake_probe_pairs(law_name, split, seeds=None):
        calls.append((law_name, split))
        return [_FakePair(f"{law_name}-{split}-{i}-obey", f"{law_name}-{split}-{i}-violate") for i in range(3)]

    monkeypatch.setattr(train_stacks, "probe_pairs", fake_probe_pairs)

    clips = train_stacks.probe_clips()

    n_laws = len(train_stacks.ALL_LAWS)
    assert len(calls) == n_laws * 2  # every (law, split) combination visited once
    assert len(clips) == n_laws * 2 * 3 * 2  # laws x splits x pairs x (obey, violate)


def test_probe_clips_real_fan_out_is_576():
    """Not monkeypatched -- exercises real probe_pairs() (pymunk) to pin the
    documented count: 6 laws (ALL_LAWS, including the eval-only laws, which
    need mechanistic probe latents too) x (32 train + 16 test) seeds x 2
    clips = 576."""
    from scripts.train_stacks import probe_clips

    clips = probe_clips()
    assert len(clips) == 576


def test_probe_caches_flag_routes_probe_clips_into_encode_clips_cached(monkeypatch, tmp_path):
    """CLI-parsing/plan-level test: --probe-caches must be accepted and must
    push probe_clips() through the same encode_clips_cached batched path as
    the regular train/val/eval clips, per selected stack. Default (flag
    absent) behaviour must stay byte-identical -- no extra encode_clips_cached
    calls beyond the legacy train/val/eval one per stack."""
    import scripts.train_stacks as train_stacks

    fake_probe_clips = ["probe-clip-a", "probe-clip-b"]
    monkeypatch.setattr(train_stacks, "probe_clips", lambda: fake_probe_clips)
    monkeypatch.setattr(train_stacks, "train_clips", lambda: [])
    monkeypatch.setattr(train_stacks, "val_clips", lambda: [])
    monkeypatch.setattr(train_stacks, "eval_pairs", lambda law_name: [])
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc, list(clips), cache_dir))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)
    monkeypatch.setattr(train_stacks.sys, "argv", ["train_stacks.py", "--stacks", "raw-pixel", "--probe-caches"])

    train_stacks.main()

    probe_calls = [c for c in encode_calls if c[1] == fake_probe_clips]
    assert len(probe_calls) == 1, (
        f"expected exactly one encode_clips_cached call carrying the probe "
        f"clips when --probe-caches is passed, got {len(probe_calls)} "
        f"(calls: {encode_calls})"
    )


class _FakeEncoder:
    def __init__(self, name):
        self.name = name
        self.cache_key = f"{name}-fakehash"


def _patch_clip_sources(monkeypatch, train_stacks):
    """Shared clip-source mocks for populate_caches causal-scope tests:
    train_clips/val_clips/probe_clips return small, distinguishable fake
    clip lists; eval_pairs(law_name) returns one MinimalPair per law whose
    obey/violate values embed the law name so callers can assert exactly
    which laws' eval clips were included."""
    monkeypatch.setattr(train_stacks, "train_clips", lambda: ["train-1", "train-2"])
    monkeypatch.setattr(train_stacks, "val_clips", lambda: ["val-1"])
    monkeypatch.setattr(train_stacks, "probe_clips", lambda: ["probe-1", "probe-2"])

    def fake_eval_pairs(law_name, seeds=None):
        return [_FakePair(f"{law_name}-obey", f"{law_name}-violate")]

    monkeypatch.setattr(train_stacks, "eval_pairs", fake_eval_pairs)


def test_causal_scope_voe_encodes_train_val_and_permanence_pairs_only(monkeypatch, tmp_path):
    """--causal-scope voe (default): a causal stack (name starting
    "vjepa2-vitl-causal") must be encoded only against train + val clips
    plus eval pairs for permanence and permanence-ext -- no other laws, and
    NEVER probe clips even when include_probe_caches=True."""
    import scripts.train_stacks as train_stacks

    _patch_clip_sources(monkeypatch, train_stacks)
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc.name, list(clips)))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)

    causal_enc = _FakeEncoder("vjepa2-vitl-causal-w16")
    train_stacks.populate_caches([causal_enc], include_probe_caches=True, causal_scope="voe")

    assert len(encode_calls) == 1
    name, clips = encode_calls[0]
    assert name == "vjepa2-vitl-causal-w16"
    assert clips == [
        "train-1", "train-2", "val-1",
        "permanence-obey", "permanence-violate",
        "permanence-ext-obey", "permanence-ext-violate",
    ]
    assert "probe-1" not in clips and "probe-2" not in clips


def test_causal_scope_full_preserves_legacy_behaviour(monkeypatch, tmp_path):
    """--causal-scope full: a causal stack gets the SAME clip set as any
    other stack -- train + val + eval pairs for every ALL_LAWS law (+ probe
    clips when include_probe_caches=True)."""
    import scripts.train_stacks as train_stacks

    _patch_clip_sources(monkeypatch, train_stacks)
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc.name, list(clips)))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)

    causal_enc = _FakeEncoder("vjepa2-vitl-causal-w32")
    train_stacks.populate_caches([causal_enc], include_probe_caches=True, causal_scope="full")

    assert len(encode_calls) == 1
    name, clips = encode_calls[0]
    assert name == "vjepa2-vitl-causal-w32"
    expected = ["train-1", "train-2", "val-1"]
    for law_name in train_stacks.ALL_LAWS:
        expected.append(f"{law_name}-obey")
        expected.append(f"{law_name}-violate")
    expected.extend(["probe-1", "probe-2"])
    assert clips == expected


def test_causal_scope_does_not_affect_non_causal_stacks(monkeypatch, tmp_path):
    """A non-causal stack (e.g. dinov2-s14) must always get the full clip
    set regardless of --causal-scope's value -- the flag only scopes stacks
    whose name starts with "vjepa2-vitl-causal"."""
    import scripts.train_stacks as train_stacks

    _patch_clip_sources(monkeypatch, train_stacks)
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc.name, list(clips)))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)

    non_causal_enc = _FakeEncoder("dinov2-s14")
    train_stacks.populate_caches([non_causal_enc], include_probe_caches=True, causal_scope="voe")

    assert len(encode_calls) == 1
    name, clips = encode_calls[0]
    assert name == "dinov2-s14"
    expected = ["train-1", "train-2", "val-1"]
    for law_name in train_stacks.ALL_LAWS:
        expected.append(f"{law_name}-obey")
        expected.append(f"{law_name}-violate")
    expected.extend(["probe-1", "probe-2"])
    assert clips == expected


def test_causal_scope_mixed_stacks_each_get_correct_clip_set(monkeypatch, tmp_path):
    """One populate_caches call with both a causal and a non-causal stack:
    each stack must see its own correctly-scoped clip set."""
    import scripts.train_stacks as train_stacks

    _patch_clip_sources(monkeypatch, train_stacks)
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc.name, list(clips)))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)

    causal_enc = _FakeEncoder("vjepa2-vitl-causal-w16")
    non_causal_enc = _FakeEncoder("dinov2-s14")
    train_stacks.populate_caches(
        [non_causal_enc, causal_enc], include_probe_caches=True, causal_scope="voe"
    )

    by_name = dict(encode_calls)
    assert by_name["vjepa2-vitl-causal-w16"] == [
        "train-1", "train-2", "val-1",
        "permanence-obey", "permanence-violate",
        "permanence-ext-obey", "permanence-ext-violate",
    ]
    full_expected = ["train-1", "train-2", "val-1"]
    for law_name in train_stacks.ALL_LAWS:
        full_expected.append(f"{law_name}-obey")
        full_expected.append(f"{law_name}-violate")
    full_expected.extend(["probe-1", "probe-2"])
    assert by_name["dinov2-s14"] == full_expected


def test_probe_caches_flag_absent_does_not_touch_probe_clips(monkeypatch, tmp_path):
    """Default behaviour (no --probe-caches) must be byte-identical to
    before: encode_clips_cached must never see the probe clips."""
    import scripts.train_stacks as train_stacks

    fake_probe_clips = ["probe-clip-a", "probe-clip-b"]
    probe_clips_called = []
    monkeypatch.setattr(train_stacks, "probe_clips", lambda: probe_clips_called.append(1) or fake_probe_clips)
    monkeypatch.setattr(train_stacks, "train_clips", lambda: [])
    monkeypatch.setattr(train_stacks, "val_clips", lambda: [])
    monkeypatch.setattr(train_stacks, "eval_pairs", lambda law_name: [])
    monkeypatch.setattr(train_stacks, "CACHE_DIR", tmp_path)

    encode_calls = []

    def fake_encode_clips_cached(enc, clips, cache_dir=None, **kwargs):
        encode_calls.append((enc, list(clips), cache_dir))
        return [None] * len(clips)

    monkeypatch.setattr(train_stacks, "encode_clips_cached", fake_encode_clips_cached)
    monkeypatch.setattr(train_stacks.sys, "argv", ["train_stacks.py", "--stacks", "raw-pixel"])

    train_stacks.main()

    assert not probe_clips_called, "probe_clips() must not be called when --probe-caches is absent"
    assert all(c[1] != fake_probe_clips for c in encode_calls)
