"""Registry of all FLOOR-suite physical laws."""
from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw

ALL_LAWS: dict[str, type] = {
    "support": SupportLaw,
    "permanence": PermanenceLaw,
    "solidity": SolidityLaw,
    "gravity": GravityLaw,
}
