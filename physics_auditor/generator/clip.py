"""A rendered rollout: frames + the states that produced them."""
from dataclasses import dataclass

import numpy as np

from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.state import WorldState


@dataclass
class Clip:
    frames: np.ndarray  # (T, 64, 64, 3) uint8
    states: list[WorldState]  # len T; states[t] is the state rendered into frames[t]
    config: ScenarioConfig
