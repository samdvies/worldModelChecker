"""Behavioural violation-of-expectation (VoE) probe."""
from dataclasses import dataclass

from physics_auditor.laws.base import MinimalPair
from physics_auditor.models.base import ModelAdapter
from physics_auditor.probes.behavioural.metrics import auroc


@dataclass
class VoEResult:
    model: str
    law: str
    auroc: float
    n_pairs: int
    obey_scores: list[float]
    violate_scores: list[float]


def run_voe(
    adapter: ModelAdapter, pairs: list[MinimalPair], law: str, horizon: int = 8
) -> VoEResult:
    obey_scores = [adapter.surprise(p.obey, p.critical_frame, horizon) for p in pairs]
    violate_scores = [adapter.surprise(p.violate, p.critical_frame, horizon) for p in pairs]
    score = auroc(violate_scores, obey_scores)
    return VoEResult(
        model=adapter.name,
        law=law,
        auroc=score,
        n_pairs=len(pairs),
        obey_scores=obey_scores,
        violate_scores=violate_scores,
    )
