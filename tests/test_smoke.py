"""Fast pre-flight smoke tier (pytest -m smoke), <= 10s wall, TORCH-FREE.

Purpose: catch import-time breakage and API-shape drift across every
torch-free physics_auditor/scripts module, plus a minimal end-to-end
exercise of the non-torch pipeline stages, BEFORE paying for a full local
suite run or a ~10min GPU-box boot.

IMPORTANT: this module (and everything it imports at collection time) must
NOT import torch, directly or transitively -- `import torch` alone measures
~15s cold on this machine (see docs/failure-sweeps.md), which blows the
<=10s budget by itself regardless of test logic. Torch-backed modules
(physics_auditor.models.encoders/predictor/pretrained and the scripts that
use them) are covered by the separate `smoke_torch` tier in
tests/test_smoke_torch.py instead -- that tier is real and required before
a GPU-box dispatch, it just isn't held to the <=10s bar.

This is deliberately not a correctness suite (see the rest of tests/ for
that) -- it is a "does the wiring still connect" suite.
"""
import importlib
import pkgutil

import numpy as np
import pytest

import physics_auditor
import scripts as scripts_pkg

pytestmark = pytest.mark.smoke

# Modules that import torch (directly or transitively) -- deliberately
# excluded from this torch-free tier and covered instead by
# tests/test_smoke_torch.py's `smoke_torch` marker. Keep in sync with the
# module list asserted by test_torch_modules_excluded_are_still_covered
# below.
TORCH_MODULES = {
    "physics_auditor.models.encoders",
    "physics_auditor.models.predictor",
    "physics_auditor.models.pretrained",
    "physics_auditor.models.stacks",  # imports LatentPredictor -> torch transitively
    "scripts.run_mechanistic",
    "scripts.run_monitor_eval",
    "scripts.run_report_card",
    "scripts.train_stacks",
}


# --------------------------------------------------------------- discovery

def _submodule_names(pkg) -> list[str]:
    return [m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}.")]


def _script_module_names() -> list[str]:
    import pathlib

    scripts_dir = pathlib.Path(scripts_pkg.__file__).parent
    names = []
    for path in sorted(scripts_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        names.append(f"scripts.{path.stem}")
    return names


_FAST_PHYSICS_AUDITOR_MODULES = [
    m for m in _submodule_names(physics_auditor) if m not in TORCH_MODULES
]
_FAST_SCRIPT_MODULES = [m for m in _script_module_names() if m not in TORCH_MODULES]


@pytest.mark.parametrize("mod_name", _FAST_PHYSICS_AUDITOR_MODULES)
def test_physics_auditor_module_imports(mod_name):
    """Every torch-free physics_auditor.* submodule must import cleanly."""
    importlib.import_module(mod_name)


@pytest.mark.parametrize("mod_name", _FAST_SCRIPT_MODULES)
def test_scripts_module_imports(mod_name):
    """Every torch-free scripts/*.py must import cleanly as a module (class A
    guard: remote_job.sh's embedded snippets aren't the only place
    import-time breakage can hide -- these are the real modules the tar
    ships)."""
    importlib.import_module(mod_name)


def test_torch_modules_excluded_are_still_covered():
    """The TORCH_MODULES exclusion list must exactly match reality: every
    name in it must actually exist as a discoverable module (no stale
    entries silently shrinking coverage), and the smoke_torch tier's own
    discovery (tests/test_smoke_torch.py) is responsible for importing all
    of them -- this just guards against TORCH_MODULES drifting out of sync
    with the real module set."""
    all_pa = set(_submodule_names(physics_auditor))
    all_scripts = set(_script_module_names())
    all_known = all_pa | all_scripts
    stale = TORCH_MODULES - all_known
    assert not stale, f"TORCH_MODULES contains modules that no longer exist: {stale}"


# ------------------------------------------------------- minimal generation

def _tiny_config(seed: int, violate: bool):
    from physics_auditor.generator.config import BodySpec, ScenarioConfig

    ball = BodySpec(
        body_id="ball",
        shape="circle",
        position=(10.0, 10.0),
        velocity=(30.0, 80.0),
        radius=3.0,
        mass=1.0,
        friction=0.0,
    )
    return ScenarioConfig(
        seed=seed,
        law="gravity",
        violate=violate,
        bodies=(ball,),
        n_frames=8,
        critical_frame=4,
        intervention="gravity_off",
        intervention_target="ball",
    )


def test_minimal_pair_prefix_identity_on_reduced_config():
    """obey/violate clips built from a REDUCED (n_frames=8) config must be
    pixel-identical up to (not including) critical_frame -- the same
    minimal-pair invariant the full-size law tests check, at 1/6th the
    frame count."""
    from physics_auditor.generator.simulate import simulate

    obey = simulate(_tiny_config(seed=0, violate=False))
    violate = simulate(_tiny_config(seed=0, violate=True))

    assert obey.frames.shape == (8, 64, 64, 3)
    np.testing.assert_array_equal(
        obey.frames[:4], violate.frames[:4]
    )
    assert not np.array_equal(obey.frames[4:], violate.frames[4:])


# -------------------------------------------------------------------- voe

class _StubAdapter:
    """Deterministic stub ModelAdapter: surprise = 1.0 if the clip is a
    violate config, else 0.0 -- perfectly separates without touching any
    real model, encoder, or on-disk cache (and, notably, without torch)."""

    name = "stub"

    def surprise(self, clip, critical_frame, horizon=8):
        return 1.0 if clip.config.violate else 0.0


def test_voe_on_stub_adapter():
    from physics_auditor.generator.simulate import simulate
    from physics_auditor.laws.base import MinimalPair
    from physics_auditor.probes.behavioural.voe import run_voe

    pairs = [
        MinimalPair(
            obey=simulate(_tiny_config(seed=s, violate=False)),
            violate=simulate(_tiny_config(seed=s, violate=True)),
            critical_frame=4,
        )
        for s in range(3)
    ]
    result = run_voe(_StubAdapter(), pairs, law="gravity")
    assert result.auroc == 1.0
    assert result.n_pairs == 3


# ----------------------------------------------------------------- report

def test_report_card_render():
    from physics_auditor.report.card import ReportCard

    card = ReportCard()
    card.add_row("stub", "gravity", "behavioural", "auroc", 0.875, 3)
    md = card.to_markdown()
    assert "stub" in md
    assert "gravity" in md
    assert "0.875" in md


# ---------------------------------------------------------------- monitor

def test_monitor_threshold_math_on_synthetic_scores():
    from physics_auditor.monitor.detector import calibrate_threshold, first_fire

    obey_scores = [np.array([0.1, 0.2, 0.3, 0.9]), np.array([0.15, 0.25])]
    threshold = calibrate_threshold(obey_scores, percentile=50.0)
    assert threshold > 0.0

    fired = first_fire(np.array([0.05, 0.05, threshold + 1.0, 0.05]), threshold, k=4)
    assert fired == 6  # k + index 2

    not_fired = first_fire(np.array([0.0, 0.0]), threshold, k=4)
    assert not_fired is None


# ------------------------------------------------------------------ gallery

def test_gallery_pure_functions_on_synthetic_input():
    from physics_auditor.gallery.traces import (
        frame_to_png_base64,
        normalize_trace_pair,
        pixel_static_one_step_trace,
    )

    class _FakeConfig:
        violate = False

    class _FakeClip:
        def __init__(self):
            self.frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
            self.config = _FakeConfig()

    clip = _FakeClip()
    trace = pixel_static_one_step_trace(clip)
    assert trace.shape == (5,)

    png_b64 = frame_to_png_base64(clip.frames[0])
    assert isinstance(png_b64, str) and len(png_b64) > 0

    obey_norm, violate_norm, normalizer = normalize_trace_pair(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0, 4.0])
    )
    assert normalizer == 2.0
    np.testing.assert_allclose(obey_norm, [0.0, 0.5, 1.0])
