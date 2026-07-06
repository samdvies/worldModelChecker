"""TDD tests for the oracle model adapter and behavioural VoE probe."""
import pytest

from physics_auditor.laws.support import SupportLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.voe import VoEResult, run_voe


@pytest.fixture(scope="module")
def pair0():
    return SupportLaw().generate_pair(0)


@pytest.fixture(scope="module")
def pairs8():
    return [SupportLaw().generate_pair(s) for s in range(8)]


def test_oracle_surprise_low_on_obey_high_on_violate(pair0):
    adapter = OracleAdapter()
    obey_score = adapter.surprise(pair0.obey, pair0.critical_frame)
    violate_score = adapter.surprise(pair0.violate, pair0.critical_frame)
    assert violate_score > 10 * obey_score, (obey_score, violate_score)


def test_oracle_closes_loop(pairs8):
    adapter = OracleAdapter()
    result = run_voe(adapter, pairs8, law="support")
    assert result.auroc >= 0.95, result.auroc


def test_voe_result_fields(pairs8):
    adapter = OracleAdapter()
    result = run_voe(adapter, pairs8, law="support")
    assert isinstance(result, VoEResult)
    assert result.n_pairs == 8
    assert len(result.obey_scores) == 8
    assert len(result.violate_scores) == 8
    assert result.model == "oracle"
    assert result.law == "support"
