"""Scenario configuration: everything needed to deterministically build a Clip."""
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BodySpec:
    body_id: str
    shape: str  # 'box' | 'circle'
    position: tuple[float, float]
    mass: float
    velocity: tuple[float, float] = (0.0, 0.0)
    half_extents: tuple[float, float] | None = None  # boxes
    radius: float | None = None  # circles
    friction: float = 0.9
    is_static: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int
    law: str
    violate: bool
    bodies: tuple[BodySpec, ...]
    n_frames: int = 48
    critical_frame: int = 24
    intervention: str | None = None
    intervention_target: str | None = None
    occluders: tuple[tuple[float, float, float, float], ...] = ()  # render-only rects (x, y, w, h)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "law": self.law,
            "violate": self.violate,
            "bodies": [asdict(b) for b in self.bodies],
            "n_frames": self.n_frames,
            "critical_frame": self.critical_frame,
            "intervention": self.intervention,
            "intervention_target": self.intervention_target,
            "occluders": [list(o) for o in self.occluders],
        }

    @property
    def scenario_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
