"""Feasibility demo: run the oracle model through the support-law VoE probe.

Usage: uv run python scripts/run_feasibility.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

from physics_auditor.laws.support import SupportLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.voe import run_voe
from physics_auditor.report.card import ReportCard

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def filmstrip(frames: np.ndarray, stride: int = 6) -> np.ndarray:
    return np.hstack(frames[::stride])


def main() -> None:
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

    print(card.to_markdown())
    print()
    print("per-pair surprise scores (obey / violate):")
    for seed, obey_score, violate_score in zip(
        range(8), result.obey_scores, result.violate_scores
    ):
        print(f"  seed={seed}: obey={obey_score:.4f}  violate={violate_score:.4f}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    pair0 = pairs[0]
    Image.fromarray(filmstrip(pair0.obey.frames)).save(ARTIFACTS_DIR / "obey.png")
    Image.fromarray(filmstrip(pair0.violate.frames)).save(ARTIFACTS_DIR / "violate.png")
    print()
    print(f"saved filmstrips to {ARTIFACTS_DIR / 'obey.png'} and {ARTIFACTS_DIR / 'violate.png'}")


if __name__ == "__main__":
    main()
