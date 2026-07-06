"""Validity tests for the Gravity law minimal-pair slice."""
import numpy as np
import pytest

from physics_auditor.generator.engine import Engine
from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.laws.gravity import GravityLaw


@pytest.fixture(scope="module")
def pair0():
    return GravityLaw().generate_pair(0)


def test_prefix_frames_identical(pair0):
    cf = pair0.critical_frame
    obey, violate = pair0.obey, pair0.violate
    for t in range(cf):
        assert np.array_equal(obey.frames[t], violate.frames[t]), f"frame {t} differs"
        assert obey.states[t] == violate.states[t], f"state {t} differs"


def test_configs_differ_in_exactly_one_variable(pair0):
    d_obey = pair0.obey.config.to_dict()
    d_violate = pair0.violate.config.to_dict()
    diff_keys = {k for k in d_obey if d_obey[k] != d_violate.get(k)}
    assert diff_keys == {"violate"}


def test_violate_actually_violates_gt(pair0):
    cf = pair0.critical_frame
    obey, violate = pair0.obey, pair0.violate

    lawful_engine = Engine.from_state(obey.states[cf - 1])
    lawful_states = []
    for _ in range(8):
        lawful_states.append(lawful_engine.step())

    obey_window = obey.states[cf:cf + 8]
    for lawful_s, obey_s in zip(lawful_states, obey_window):
        lb = next(b for b in lawful_s.bodies if b.body_id == "ball")
        ob = next(b for b in obey_s.bodies if b.body_id == "ball")
        err = ((lb.position[0] - ob.position[0]) ** 2 + (lb.position[1] - ob.position[1]) ** 2) ** 0.5
        assert err < 0.5, f"lawful vs obey diverged too much: {err}"

    violate_window = violate.states[cf:cf + 8]
    summed_err = 0.0
    for lawful_s, violate_s in zip(lawful_states, violate_window):
        lb = next(b for b in lawful_s.bodies if b.body_id == "ball")
        vb = next(b for b in violate_s.bodies if b.body_id == "ball")
        summed_err += ((lb.position[0] - vb.position[0]) ** 2 + (lb.position[1] - vb.position[1]) ** 2) ** 0.5
    assert summed_err > 2.0, f"lawful rollout too close to violate: {summed_err}"


def test_obey_parabola_vy_strictly_decreasing_while_airborne(pair0):
    cf = pair0.critical_frame
    obey = pair0.obey
    vy = [next(b for b in s.bodies if b.body_id == "ball").velocity[1] for s in obey.states[cf:cf + 9]]
    for a, b in zip(vy, vy[1:]):
        assert b < a, f"vy did not strictly decrease: {vy}"


def test_obey_ball_airborne_through_critical_plus_8_all_seeds():
    for seed in range(8):
        pair = GravityLaw().generate_pair(seed)
        cf = pair.critical_frame
        obey = pair.obey
        for t in range(cf, cf + 9):
            state = obey.states[t]
            ball = next(b for b in state.bodies if b.body_id == "ball")
            assert ball.position[1] > GROUND_Y + 3.0 + 1.0, f"seed {seed} frame {t}: ball not airborne"
            assert not any("ball" in edge for edge in state.support_edges), (
                f"seed {seed} frame {t}: ball has support edges"
            )


def test_violate_velocity_stays_constant(pair0):
    cf = pair0.critical_frame
    violate = pair0.violate
    velocities = [next(b for b in s.bodies if b.body_id == "ball").velocity for s in violate.states[cf:cf + 9]]
    for (vx_a, vy_a), (vx_b, vy_b) in zip(velocities, velocities[1:]):
        dv = ((vx_b - vx_a) ** 2 + (vy_b - vy_a) ** 2) ** 0.5
        assert dv < 0.5, f"velocity changed under gravity_off: {velocities}"


def test_violate_diverges_from_obey_vy_drop(pair0):
    cf = pair0.critical_frame
    obey = pair0.obey
    vy_obey = [next(b for b in s.bodies if b.body_id == "ball").velocity[1] for s in obey.states[cf:cf + 9]]
    drop = vy_obey[0] - vy_obey[-1]
    assert drop > 40.0, f"obey vy did not drop as expected: {vy_obey}"


def test_clip_shapes_and_types(pair0):
    obey = pair0.obey
    assert obey.frames.shape == (48, 64, 64, 3)
    assert obey.frames.dtype == np.uint8
    assert len(obey.states) == 48


def test_determinism_same_seed():
    pair_a = GravityLaw().generate_pair(0)
    pair_b = GravityLaw().generate_pair(0)
    assert np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
    assert np.array_equal(pair_a.violate.frames, pair_b.violate.frames)


def test_seeds_differ():
    pair_a = GravityLaw().generate_pair(0)
    pair_b = GravityLaw().generate_pair(1)
    assert not np.array_equal(pair_a.obey.frames, pair_b.obey.frames)


@pytest.mark.parametrize("seed", range(8))
def test_stability_across_seeds(seed):
    pair = GravityLaw().generate_pair(seed)
    cf = pair.critical_frame
    obey, violate = pair.obey, pair.violate
    for t in range(cf):
        assert np.array_equal(obey.frames[t], violate.frames[t])
    vy_obey = [next(b for b in s.bodies if b.body_id == "ball").velocity[1] for s in obey.states[cf:cf + 9]]
    for a, b in zip(vy_obey, vy_obey[1:]):
        assert b < a
    vel_violate = [next(b for b in s.bodies if b.body_id == "ball").velocity for s in violate.states[cf:cf + 9]]
    for (vx_a, vy_a), (vx_b, vy_b) in zip(vel_violate, vel_violate[1:]):
        dv = ((vx_b - vx_a) ** 2 + (vy_b - vy_a) ** 2) ** 0.5
        assert dv < 0.5
