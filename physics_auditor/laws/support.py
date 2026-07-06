"""Support/stability law: a 3-block tower that either stays put or collapses."""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair

WORLD_CENTRE_X = 32.0


class SupportLaw:
    name = "support"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        widths = tuple(float(w) for w in rng.uniform(8, 14, size=3))
        heights = tuple(float(h) for h in rng.uniform(5, 8, size=3))
        tower_offset = float(rng.uniform(-4, 4))
        jitters = rng.uniform(-1, 1, size=3)
        x_centres = tuple(float(WORLD_CENTRE_X + tower_offset + j) for j in jitters)

        obey_config = ScenarioConfig(
            seed=seed,
            violate=False,
            widths=widths,
            heights=heights,
            x_centres=x_centres,
        )
        violate_config = replace(obey_config, violate=True)

        obey_clip = simulate(obey_config)
        violate_clip = simulate(violate_config)

        return MinimalPair(
            obey=obey_clip,
            violate=violate_clip,
            critical_frame=obey_config.critical_frame,
            differing_variable="violate",
        )
