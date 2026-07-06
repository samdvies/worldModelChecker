"""Solidity law: a ball dropped onto a shelf that may (or may not) be ghosted."""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair

SHELF_Y = 20.0
SHELF_HALF_EXTENTS = (12.0, 1.0)
BALL_RADIUS = 3.0
BALL_START_Y = 50.0
CRITICAL_FRAME = 14


class SolidityLaw:
    name = "solidity"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        shelf_x = float(32.0 + rng.uniform(-2, 2))
        ball_x = float(shelf_x + rng.uniform(-4, 4))

        shelf = BodySpec(
            body_id="shelf",
            shape="box",
            position=(shelf_x, SHELF_Y),
            mass=1.0,
            half_extents=SHELF_HALF_EXTENTS,
            is_static=True,
        )
        ball = BodySpec(
            body_id="ball",
            shape="circle",
            position=(ball_x, BALL_START_Y),
            velocity=(0.0, 0.0),
            radius=BALL_RADIUS,
            mass=1.0,
        )

        obey_config = ScenarioConfig(
            seed=seed,
            law=self.name,
            violate=False,
            bodies=(shelf, ball),
            critical_frame=CRITICAL_FRAME,
            intervention="ghost",
            intervention_target="shelf",
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
