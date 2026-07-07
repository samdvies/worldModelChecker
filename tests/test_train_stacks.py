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
