"""Validity tests for the Solidity law minimal-pair slice."""
import numpy as np
import pytest

from physics_auditor.generator.engine import Engine
from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.laws.solidity import SHELF_HALF_EXTENTS, SolidityLaw


@pytest.fixture(scope="module")
def pair0():
    return SolidityLaw().generate_pair(0)


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
        for lb, ob in zip(lawful_s.bodies, obey_s.bodies):
            assert lb.body_id == ob.body_id
            err = ((lb.position[0] - ob.position[0]) ** 2 + (lb.position[1] - ob.position[1]) ** 2) ** 0.5
            assert err < 0.5, f"lawful vs obey diverged too much: {err}"

    violate_window = violate.states[cf:cf + 8]
    summed_err = 0.0
    for lawful_s, violate_s in zip(lawful_states, violate_window):
        for lb, vb in zip(lawful_s.bodies, violate_s.bodies):
            summed_err += ((lb.position[0] - vb.position[0]) ** 2 + (lb.position[1] - vb.position[1]) ** 2) ** 0.5
    assert summed_err > 2.0, f"lawful rollout too close to violate: {summed_err}"


def test_ball_well_above_shelf_at_critical_frame_all_seeds():
    for seed in range(8):
        pair = SolidityLaw().generate_pair(seed)
        cf = pair.critical_frame
        state = pair.obey.states[cf]
        ball = next(b for b in state.bodies if b.body_id == "ball")
        shelf = next(b for b in state.bodies if b.body_id == "shelf")
        shelf_top = shelf.position[1] + SHELF_HALF_EXTENTS[1]
        gap = (ball.position[1] - ball.radius) - shelf_top
        assert gap > 2.0, f"seed {seed}: ball only {gap} above shelf at critical frame"


def test_obey_ball_lands_on_shelf_and_stays_above_it():
    for seed in range(8):
        pair = SolidityLaw().generate_pair(seed)
        obey = pair.obey
        shelf = next(b for b in obey.states[-1].bodies if b.body_id == "shelf")
        shelf_top = shelf.position[1] + SHELF_HALF_EXTENTS[1]
        ball_final = next(b for b in obey.states[-1].bodies if b.body_id == "ball")
        assert ball_final.position[1] > shelf_top, f"seed {seed}: ball not resting above shelf"


def test_violate_ball_passes_through_shelf_to_ground():
    for seed in range(8):
        pair = SolidityLaw().generate_pair(seed)
        violate = pair.violate
        shelf = next(b for b in violate.states[-1].bodies if b.body_id == "shelf")
        shelf_bottom = shelf.position[1] - SHELF_HALF_EXTENTS[1]
        ball_final = next(b for b in violate.states[-1].bodies if b.body_id == "ball")
        assert ball_final.position[1] < shelf_bottom - 2.0, f"seed {seed}: ball did not fall through shelf"
        assert ball_final.position[1] < GROUND_Y + ball_final.radius + 1.0, f"seed {seed}: ball not on ground"


def test_clip_shapes_and_types(pair0):
    obey = pair0.obey
    assert obey.frames.shape == (48, 64, 64, 3)
    assert obey.frames.dtype == np.uint8
    assert len(obey.states) == 48


def test_determinism_same_seed():
    pair_a = SolidityLaw().generate_pair(0)
    pair_b = SolidityLaw().generate_pair(0)
    assert np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
    assert np.array_equal(pair_a.violate.frames, pair_b.violate.frames)


def test_seeds_differ():
    pair_a = SolidityLaw().generate_pair(0)
    pair_b = SolidityLaw().generate_pair(1)
    assert not np.array_equal(pair_a.obey.frames, pair_b.obey.frames)


def test_stability_across_seeds():
    for seed in range(8):
        pair = SolidityLaw().generate_pair(seed)
        assert pair.obey.frames.shape == (48, 64, 64, 3)
        assert pair.violate.frames.shape == (48, 64, 64, 3)
