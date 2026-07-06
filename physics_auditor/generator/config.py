"""Scenario configuration: everything needed to deterministically build a Clip."""
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int
    violate: bool
    widths: tuple[float, ...]
    heights: tuple[float, ...]
    x_centres: tuple[float, ...]
    n_frames: int = 48
    critical_frame: int = 24

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "violate": self.violate,
            "widths": list(self.widths),
            "heights": list(self.heights),
            "x_centres": list(self.x_centres),
            "n_frames": self.n_frames,
            "critical_frame": self.critical_frame,
        }

    @property
    def scenario_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
