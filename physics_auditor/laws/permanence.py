"""Permanence law: a rolling ball passes behind an occluder and either
re-emerges (obey) or is deleted while hidden (violate)."""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair

RADIUS = 3.0
# Occluder wide enough that, for the full parameter range below, the ball is
# fully hidden at critical_frame AND stays hidden for several frames after
# (needed so violate's deletion is pixel-invisible at the moment it happens).
OCCLUDER = (24.0, 4.0, 20.0, 10.0)  # (x, y, w, h)


class PermanenceLaw:
    name = "permanence"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        x0 = float(8.0 + rng.uniform(-2, 2))
        vx = float(30.0 + rng.uniform(-3, 3))
        y0 = GROUND_Y + RADIUS

        ball = BodySpec(
            body_id="ball",
            shape="circle",
            position=(x0, y0),
            velocity=(vx, 0.0),
            radius=RADIUS,
            mass=1.0,
            friction=0.0,
        )

        obey_config = ScenarioConfig(
            seed=seed,
            law=self.name,
            violate=False,
            bodies=(ball,),
            intervention="delete",
            intervention_target="ball",
            occluders=(OCCLUDER,),
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
