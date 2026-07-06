"""Drive an Engine through a ScenarioConfig to produce a Clip."""
import numpy as np

from physics_auditor.generator.clip import Clip
from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.engine import Engine
from physics_auditor.generator.render import render

_INTERVENTIONS = {
    "ghost": Engine.ghost,
    "delete": Engine.delete,
    "gravity_off": Engine.gravity_off,
}


def simulate(config: ScenarioConfig) -> Clip:
    engine = Engine(config)
    states = [engine.state]
    frames = [render(states[0], config.occluders)]

    for t in range(1, config.n_frames):
        if config.violate and t == config.critical_frame and config.intervention:
            apply = _INTERVENTIONS[config.intervention]
            apply(engine, config.intervention_target)
        state = engine.step()
        states.append(state)
        frames.append(render(state, config.occluders))

    return Clip(frames=np.stack(frames), states=states, config=config)
