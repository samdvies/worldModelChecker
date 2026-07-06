"""End-to-end feasibility test: 8 pairs, oracle adapter, report card."""
from physics_auditor.laws.support import SupportLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.voe import run_voe
from physics_auditor.report.card import ReportCard


def test_feasibility_loop_closes():
    pairs = [SupportLaw().generate_pair(seed) for seed in range(8)]
    adapter = OracleAdapter()
    result = run_voe(adapter, pairs, law="support")

    card = ReportCard()
    card.add_row(
        model=result.model,
        law=result.law,
        probe="behavioural-voe",
        metric="auroc",
        value=result.auroc,
        n_pairs=result.n_pairs,
    )

    assert "oracle" in card.to_markdown()
    assert result.auroc >= 0.95
