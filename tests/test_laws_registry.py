"""TDD: laws/__init__.py exposes an ALL_LAWS registry keyed by law name."""
from physics_auditor.laws import ALL_LAWS
from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw


def test_all_laws_has_exactly_the_four_floor_laws():
    assert set(ALL_LAWS.keys()) == {"support", "permanence", "solidity", "gravity"}


def test_all_laws_maps_to_correct_classes():
    assert ALL_LAWS["support"] is SupportLaw
    assert ALL_LAWS["permanence"] is PermanenceLaw
    assert ALL_LAWS["solidity"] is SolidityLaw
    assert ALL_LAWS["gravity"] is GravityLaw
