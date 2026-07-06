"""Gravity law: a projectile ball either keeps falling or freezes mid-air."""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair


class GravityLaw:
    name = "gravity"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        x0 = float(10 + rng.uniform(-2, 2))
        vx0 = float(30 + rng.uniform(-2, 2))
        vy0 = float(80 + rng.uniform(-4, 4))

        ball = BodySpec(
            body_id="ball",
            shape="circle",
            position=(x0, 10.0),
            velocity=(vx0, vy0),
            radius=3.0,
            mass=1.0,
            friction=0.0,
        )

        obey_config = ScenarioConfig(
            seed=seed,
            law=self.name,
            violate=False,
            bodies=(ball,),
            critical_frame=12,
            intervention="gravity_off",
            intervention_target="ball",
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
