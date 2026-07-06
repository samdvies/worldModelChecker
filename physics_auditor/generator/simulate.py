"""Drive an Engine through a ScenarioConfig to produce a Clip."""
import numpy as np

from physics_auditor.generator.clip import Clip
from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.engine import Engine
from physics_auditor.generator.render import render


def simulate(config: ScenarioConfig) -> Clip:
    engine = Engine(config)
    states = [engine.state]
    frames = [render(states[0])]

    for t in range(1, config.n_frames):
        if config.violate and t == config.critical_frame:
            engine.ghost("block1")
        state = engine.step()
        states.append(state)
        frames.append(render(state))

    return Clip(frames=np.stack(frames), states=states, config=config)
