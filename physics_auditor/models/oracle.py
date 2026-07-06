"""Oracle model adapter: a perfect physics simulator used to close the feasibility loop."""
from physics_auditor.generator.clip import Clip
from physics_auditor.generator.engine import Engine
from physics_auditor.generator.state import state_distance


class OracleAdapter:
    name = "oracle"

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float:
        engine = Engine.from_state(clip.states[critical_frame - 1])
        max_horizon = min(horizon, len(clip.states) - critical_frame)
        actual_window = clip.states[critical_frame:critical_frame + max_horizon]

        total_error = 0.0
        for actual_state in actual_window:
            predicted_state = engine.step()
            total_error += state_distance(predicted_state, actual_state)
        return total_error
