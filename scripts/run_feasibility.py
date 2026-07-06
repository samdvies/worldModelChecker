"""Oracle validity sweep: run the oracle model through the VoE probe for all 4 FLOOR laws.

Usage: uv run python scripts/run_feasibility.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.voe import run_voe
from physics_auditor.report.card import ReportCard

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

LAWS = [SupportLaw, PermanenceLaw, SolidityLaw, GravityLaw]


def filmstrip(frames: np.ndarray, stride: int = 6) -> np.ndarray:
    return np.hstack(frames[::stride])


def main() -> None:
    adapter = OracleAdapter()
    card = ReportCard()
    results = {}

    for law_cls in LAWS:
        law = law_cls()
        pairs = [law.generate_pair(seed) for seed in range(8)]
        result = run_voe(adapter, pairs, law=law.name)
        results[law.name] = (result, pairs)
        card.add_row(
            model=result.model,
            law=result.law,
            probe="behavioural-voe",
            metric="auroc",
            value=result.auroc,
            n_pairs=result.n_pairs,
        )

    print(card.to_markdown())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    for law_name, (result, pairs) in results.items():
        print()
        print(f"[{law_name}] per-pair surprise scores (obey / violate):")
        for seed, obey_score, violate_score in zip(
            range(8), result.obey_scores, result.violate_scores
        ):
            print(f"  seed={seed}: obey={obey_score:.4f}  violate={violate_score:.4f}")

        pair0 = pairs[0]
        Image.fromarray(filmstrip(pair0.obey.frames)).save(ARTIFACTS_DIR / f"{law_name}_obey.png")
        Image.fromarray(filmstrip(pair0.violate.frames)).save(ARTIFACTS_DIR / f"{law_name}_violate.png")
        print(f"  saved filmstrips to {law_name}_obey.png and {law_name}_violate.png")


if __name__ == "__main__":
    main()
