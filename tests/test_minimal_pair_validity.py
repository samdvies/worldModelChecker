"""Validity tests for the World Generator + Support law minimal-pair slice."""
import numpy as np
import pytest

from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.engine import Engine
from physics_auditor.laws.support import SupportLaw


@pytest.fixture(scope="module")
def pair0():
    return SupportLaw().generate_pair(0)


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


def test_violate_breaks_support(pair0):
    cf = pair0.critical_frame
    obey, violate = pair0.obey, pair0.violate
    for t in range(cf + 1, cf + 6):
        v_edges = violate.states[t].support_edges
        assert not any(u == "block2" for u, _ in v_edges), f"violate still supports block2 at frame {t}"
        o_edges = obey.states[t].support_edges
        assert ("block2", "block1") in o_edges, f"obey missing block2/block1 support at frame {t}"


def test_obey_tower_is_stable(pair0):
    obey = pair0.obey
    first = obey.states[0]
    last = obey.states[-1]
    for fb, lb in zip(first.bodies, last.bodies):
        disp = ((fb.position[0] - lb.position[0]) ** 2 + (fb.position[1] - lb.position[1]) ** 2) ** 0.5
        assert disp < 1.0, f"{fb.body_id} displaced {disp}"


def test_violate_tower_collapses(pair0):
    cf = pair0.critical_frame
    violate = pair0.violate
    y_at_critical = next(b.position[1] for b in violate.states[cf].bodies if b.body_id == "block2")
    y_at_final = next(b.position[1] for b in violate.states[-1].bodies if b.body_id == "block2")
    assert (y_at_critical - y_at_final) > 3.0


def test_clip_shapes_and_types(pair0):
    obey = pair0.obey
    assert obey.frames.shape == (48, 64, 64, 3)
    assert obey.frames.dtype == np.uint8
    assert len(obey.states) == 48


def test_determinism_same_seed():
    pair_a = SupportLaw().generate_pair(0)
    pair_b = SupportLaw().generate_pair(0)
    assert np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
    assert np.array_equal(pair_a.violate.frames, pair_b.violate.frames)


def test_seeds_differ():
    pair_a = SupportLaw().generate_pair(0)
    pair_b = SupportLaw().generate_pair(1)
    assert not np.array_equal(pair_a.obey.frames, pair_b.obey.frames)


def test_obey_tower_settles_without_construction_drift(pair0):
    """Blocks should start in true contact equilibrium (no ground-radius interpenetration
    artifact). Bottom of block0 should be at GROUND_Y at frame 0, not drift upward as the
    engine resolves an initial overlap with the ground segment's collision radius."""
    obey = pair0.obey
    from physics_auditor.generator.physics import GROUND_Y

    block0_frame0 = next(b for b in obey.states[0].bodies if b.body_id == "block0")
    bottom_y_frame0 = block0_frame0.position[1] - block0_frame0.half_extents[1]
    assert abs(bottom_y_frame0 - GROUND_Y) < 1e-6, (
        f"block0 bottom at frame 0 is {bottom_y_frame0}, expected exactly GROUND_Y={GROUND_Y}"
    )

    block0_frame23 = next(b for b in obey.states[23].bodies if b.body_id == "block0")
    drift = abs(block0_frame23.position[1] - block0_frame0.position[1])
    assert drift < 0.05, f"block0 drifted {drift} units before critical frame (construction settling artifact)"
