"""TDD: laws/__init__.py exposes an ALL_LAWS registry keyed by law name,
plus a TRAIN_LAWS subset that excludes eval-only laws.

TRAIN_LAWS must stay exactly the original four FLOOR-suite laws: it is what
physics_auditor/generator/dataset.py's train_clips()/val_clips() must
iterate over (instead of ALL_LAWS) so that eval-only laws never enter the
predictor/encoder training distribution. ALL_LAWS is the superset consumed
by eval_pairs/probe_pairs/the report card/the gallery/mechanistic data.
"""
from physics_auditor.laws import ALL_LAWS, TRAIN_LAWS
from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.permanence_ext import PermanenceExtLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw
from physics_auditor.laws.support_hard import SupportHardLaw

ORIGINAL_FOUR = {"support", "permanence", "solidity", "gravity"}


def test_train_laws_is_exactly_the_original_four_floor_laws():
    assert set(TRAIN_LAWS.keys()) == ORIGINAL_FOUR


def test_train_laws_maps_to_correct_classes():
    assert TRAIN_LAWS["support"] is SupportLaw
    assert TRAIN_LAWS["permanence"] is PermanenceLaw
    assert TRAIN_LAWS["solidity"] is SolidityLaw
    assert TRAIN_LAWS["gravity"] is GravityLaw


def test_all_laws_contains_the_original_four_plus_eval_only_laws():
    assert set(ALL_LAWS.keys()) == ORIGINAL_FOUR | {"permanence-ext", "support-hard"}


def test_all_laws_maps_to_correct_classes():
    assert ALL_LAWS["support"] is SupportLaw
    assert ALL_LAWS["permanence"] is PermanenceLaw
    assert ALL_LAWS["solidity"] is SolidityLaw
    assert ALL_LAWS["gravity"] is GravityLaw
    assert ALL_LAWS["permanence-ext"] is PermanenceExtLaw
    assert ALL_LAWS["support-hard"] is SupportHardLaw


def test_eval_only_laws_are_in_all_laws_but_not_train_laws():
    for name in ("permanence-ext", "support-hard"):
        assert name in ALL_LAWS
        assert name not in TRAIN_LAWS
