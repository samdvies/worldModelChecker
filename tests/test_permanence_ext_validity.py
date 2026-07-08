"""Validity tests for the EVAL-ONLY Permanence-ext law minimal-pair slice.

Permanence-ext exists to fix a structurally-blind-window problem in the
base Permanence law: the violation only becomes pixel-visible when the ball
should re-emerge, which in the base law happens ~9 frames before the clip
ends, leaving almost no post-violation evidence. Here the geometry is
derived (see physics_auditor/laws/permanence_ext.py module docstring) so
the ball re-emerges well before the clip ends, leaving a large evidence
window.
"""
import numpy as np
import pytest

from physics_auditor.generator.engine import Engine
from physics_auditor.generator.state import state_distance
from physics_auditor.laws.permanence_ext import OCCLUDER, RADIUS, PermanenceExtLaw


@pytest.fixture(scope="module")
def pair0():
    return PermanenceExtLaw().generate_pair(0)


def test_name():
    assert PermanenceExtLaw().name == "permanence-ext"


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
    obey_err = sum(state_distance(lawful_s, obey_s) for lawful_s, obey_s in zip(lawful_states, obey_window))
    assert obey_err < 1.0, f"lawful vs obey diverged too much: {obey_err}"

    violate_window = violate.states[cf:cf + 8]
    violate_err = sum(state_distance(lawful_s, violate_s) for lawful_s, violate_s in zip(lawful_states, violate_window))
    assert violate_err > 40.0, f"lawful rollout too close to violate: {violate_err}"


def test_full_occlusion_at_critical_all_seeds():
    occ_left, occ_y, occ_w, occ_h = OCCLUDER
    occ_right = occ_left + occ_w
    for seed in range(16):
        pair = PermanenceExtLaw().generate_pair(seed)
        cf = pair.critical_frame
        ball = next(b for b in pair.obey.states[cf].bodies if b.body_id == "ball")
        x = ball.position[0]
        assert x - RADIUS >= occ_left, f"seed {seed}: ball left edge {x - RADIUS} outside occluder {occ_left}"
        assert x + RADIUS <= occ_right, f"seed {seed}: ball right edge {x + RADIUS} outside occluder {occ_right}"


def test_obey_reemerges_by_frame_34_with_25_frames_to_spare_all_seeds():
    """The whole point of the -ext variant: re-emergence must happen early
    enough (<=34) that a behavioural probe scoring a horizon window after
    critical_frame has substantial post-violation evidence (>=25 frames
    remaining), unlike the base Permanence law."""
    occ_left, occ_y, occ_w, occ_h = OCCLUDER
    occ_right = occ_left + occ_w
    for seed in range(16):
        pair = PermanenceExtLaw().generate_pair(seed)
        obey = pair.obey
        xs = [next(b.position[0] for b in s.bodies if b.body_id == "ball") for s in obey.states]
        reemerge_frame = next(
            (t for t, x in enumerate(xs) if x - RADIUS > occ_right), None
        )
        assert reemerge_frame is not None, f"seed {seed}: ball never re-emerges"
        assert reemerge_frame <= 34, f"seed {seed}: re-emerges too late at frame {reemerge_frame}"
        remaining = len(xs) - reemerge_frame
        assert remaining >= 25, f"seed {seed}: only {remaining} frames remain after re-emergence"


def test_violate_ball_absent_after_critical(pair0):
    cf = pair0.critical_frame
    violate = pair0.violate
    for t in range(cf, len(violate.states)):
        assert all(b.body_id != "ball" for b in violate.states[t].bodies), f"ball still present at frame {t}"


def test_obey_ball_present_all_frames_and_reemerges(pair0):
    obey = pair0.obey
    occ_left, occ_y, occ_w, occ_h = OCCLUDER
    occ_right = occ_left + occ_w
    for t in range(len(obey.states)):
        assert any(b.body_id == "ball" for b in obey.states[t].bodies), f"ball missing at frame {t}"
    reemerged = any(
        next(b for b in s.bodies if b.body_id == "ball").position[0] > occ_right + RADIUS
        for s in obey.states
    )
    assert reemerged, "ball never re-emerges past the occluder"


def test_frames_identical_window_after_critical(pair0):
    cf = pair0.critical_frame
    obey, violate = pair0.obey, pair0.violate
    for t in range(cf, cf + 4):
        assert np.array_equal(obey.frames[t], violate.frames[t]), f"frame {t} differs post-critical"


def test_clip_shapes_and_types(pair0):
    obey = pair0.obey
    assert obey.frames.shape == (64, 64, 64, 3)
    assert obey.frames.dtype == np.uint8
    assert len(obey.states) == 64


def test_determinism_same_seed():
    pair_a = PermanenceExtLaw().generate_pair(0)
    pair_b = PermanenceExtLaw().generate_pair(0)
    assert np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
    assert np.array_equal(pair_a.violate.frames, pair_b.violate.frames)


def test_seeds_differ():
    pair_a = PermanenceExtLaw().generate_pair(0)
    pair_b = PermanenceExtLaw().generate_pair(1)
    assert not np.array_equal(pair_a.obey.frames, pair_b.obey.frames)


def test_stability_across_seeds():
    for seed in range(8):
        pair = PermanenceExtLaw().generate_pair(seed)
        assert pair.obey.frames.shape == (64, 64, 64, 3)
        assert pair.violate.frames.shape == (64, 64, 64, 3)
