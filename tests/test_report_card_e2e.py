"""E2E smoke test for scripts/run_report_card.py: build all 6 adapters, run
VoE on a tiny slice (2 pairs, 1 law), assert finite scores and a rendered
card -- without paying for the full 16x4x6 grid."""
import math

from physics_auditor.generator.dataset import eval_pairs
from physics_auditor.probes.behavioural.voe import run_voe
from physics_auditor.report.card import ReportCard
from scripts.run_report_card import build_adapters


def test_all_adapters_produce_finite_scores_and_card_renders():
    adapters = build_adapters()
    assert len(adapters) == 6

    pairs = eval_pairs("support", seeds=range(0, 2))
    assert len(pairs) == 2

    card = ReportCard()
    for adapter in adapters:
        result = run_voe(adapter, pairs, "support")
        assert math.isfinite(result.auroc)
        for s in result.obey_scores + result.violate_scores:
            assert math.isfinite(s)
        card.add_row(adapter.name, "support", "behavioural", "auroc", result.auroc, result.n_pairs)

    md = card.to_markdown()
    assert "support" in md
    assert "oracle" in md
    assert len(card.rows) == 6
