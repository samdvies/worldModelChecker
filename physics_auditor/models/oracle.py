"""Oracle model adapter: a perfect physics simulator used to close the feasibility loop."""
from physics_auditor.generator.clip import Clip
from physics_auditor.generator.engine import Engine


class OracleAdapter:
    name = "oracle"

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float:
        engine = Engine.from_state(clip.states[critical_frame - 1])
        max_horizon = min(horizon, len(clip.states) - critical_frame)
        actual_window = clip.states[critical_frame:critical_frame + max_horizon]

        total_error = 0.0
        for actual_state in actual_window:
            predicted_state = engine.step()
            actual_by_id = {b.body_id: b for b in actual_state.bodies}
            for pb in predicted_state.bodies:
                ab = actual_by_id[pb.body_id]
                total_error += (
                    (pb.position[0] - ab.position[0]) ** 2
                    + (pb.position[1] - ab.position[1]) ** 2
                ) ** 0.5
        return total_error
