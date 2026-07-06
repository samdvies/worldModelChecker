"""TDD: oracle AUROC >= 0.95 across all four FLOOR-suite laws."""
import pytest

from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.voe import run_voe

LAWS = [SupportLaw, PermanenceLaw, SolidityLaw, GravityLaw]


@pytest.mark.parametrize("law_cls", LAWS, ids=[l.name for l in LAWS])
def test_oracle_auroc_per_law(law_cls):
    law = law_cls()
    pairs = [law.generate_pair(seed) for seed in range(8)]
    adapter = OracleAdapter()
    result = run_voe(adapter, pairs, law=law.name)
    assert result.auroc >= 0.95, (law.name, result.auroc, result.obey_scores, result.violate_scores)
