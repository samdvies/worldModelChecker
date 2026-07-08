"""Validity tests for the EVAL-ONLY Support-hard law minimal-pair slice.

Support-hard exists to defeat the pixel-static baseline's free ride on the
base Support law (where the ONLY motion in either clip is the tower's own
collapse, itself only present in violate -- trivially separable by any
pixel-difference statistic with no physics understanding at all). It adds a
second, unrelated "decoy" tower that collapses under pure, unforced gravity
identically in both obey and violate (no intervention ever targets it), so
pixel-difference statistics see motion in both classes while the oracle
(which just has to get ordinary gravity right) is untouched.
"""
import math

import numpy as np
import pytest

from physics_auditor.generator.engine import Engine
from physics_auditor.laws.support_hard import SupportHardLaw
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.probes.behavioural.metrics import auroc
from physics_auditor.probes.behavioural.pixel_baseline import PixelStaticAdapter
from physics_auditor.probes.behavioural.voe import run_voe

EVAL_SEEDS = range(16)


def _rotated_corners(cx, cy, hx, hy, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
        for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    ]


def _aabb_x(body):
    hx, hy = body.half_extents
    corners = _rotated_corners(body.position[0], body.position[1], hx, hy, body.angle)
    xs = [c[0] for c in corners]
    return min(xs), max(xs)


@pytest.fixture(scope="module")
def pair0():
    return SupportHardLaw().generate_pair(0)


@pytest.fixture(scope="module")
def eval_pairs():
    return [SupportHardLaw().generate_pair(seed) for seed in EVAL_SEEDS]


def test_name():
    assert SupportHardLaw().name == "support-hard"


# --- (a) minimal-pair discipline: obey/violate byte-identical strictly
# before critical_frame, and configs differ in exactly one variable. ---


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
    lawful_states = [lawful_engine.step() for _ in range(8)]

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


# --- (b) oracle separates: AUROC == 1.0 (oracle only needs ordinary
# gravity to predict the decoy, so it is untouched by the decoy). ---


def test_oracle_auroc_is_1(eval_pairs):
    adapter = OracleAdapter()
    result = run_voe(adapter, eval_pairs, law="support-hard")
    assert result.auroc == 1.0, (result.auroc, result.obey_scores, result.violate_scores)


# --- (c) pixel-static AUROC materially below 1.0 (<=0.85) while the oracle
# stays perfect: this is the whole point of the decoy. ---


def test_pixel_static_auroc_degraded_but_oracle_perfect(eval_pairs):
    pixel_adapter = PixelStaticAdapter()
    oracle_adapter = OracleAdapter()

    pixel_result = run_voe(pixel_adapter, eval_pairs, law="support-hard")
    oracle_result = run_voe(oracle_adapter, eval_pairs, law="support-hard")

    assert pixel_result.auroc <= 0.85, (
        f"pixel-static floor not degraded enough: {pixel_result.auroc}"
    )
    assert oracle_result.auroc == 1.0, oracle_result.auroc


# --- (d) decoy tower actually collapses in both clips, verified via engine
# state (not pixels), and identically in obey vs violate. ---


def test_decoy_collapses_in_both_clips_via_engine_state(pair0):
    obey, violate = pair0.obey, pair0.violate

    def drop(clip):
        y0 = next(b.position[1] for b in clip.states[0].bodies if b.body_id == "decoyB")
        y_final = next(b.position[1] for b in clip.states[-1].bodies if b.body_id == "decoyB")
        return y0 - y_final

    obey_drop = drop(obey)
    violate_drop = drop(violate)
    assert obey_drop > 20.0, f"decoy did not collapse in obey clip (drop={obey_drop})"
    assert violate_drop > 20.0, f"decoy did not collapse in violate clip (drop={violate_drop})"
    # Its dynamics never read config.violate, so obey and violate should be
    # (near-)identical -- this is what keeps the minimal-pair discipline intact.
    assert abs(obey_drop - violate_drop) < 1e-6


# --- world-bounds / non-interpenetration sanity: every body (Tower A and
# the decoy) stays inside the 64-unit world with >=1 unit of clearance from
# walls and from each other, across the full clip and both branches. ---


def test_all_bodies_stay_in_world_bounds_with_margin(eval_pairs):
    for pair in eval_pairs:
        for clip in (pair.obey, pair.violate):
            for state in clip.states:
                for body in state.bodies:
                    left, right = _aabb_x(body)
                    assert left >= 1.0, f"{body.body_id} left edge {left} out of bounds"
                    assert right <= 63.0, f"{body.body_id} right edge {right} out of bounds"


def test_decoy_and_tower_never_interpenetrate(eval_pairs):
    for pair in eval_pairs:
        for clip in (pair.obey, pair.violate):
            for state in clip.states:
                decoy = [b for b in state.bodies if b.body_id in ("decoyA", "decoyB")]
                tower = [b for b in state.bodies if b.body_id.startswith("block")]
                decoy_right = max(_aabb_x(b)[1] for b in decoy)
                tower_left = min(_aabb_x(b)[0] for b in tower)
                assert tower_left - decoy_right >= 1.0, "decoy tower overlaps support tower"


def test_clip_shapes_and_types(pair0):
    obey = pair0.obey
    assert obey.frames.shape == (48, 64, 64, 3)
    assert obey.frames.dtype == np.uint8
    assert len(obey.states) == 48


def test_determinism_same_seed():
    pair_a = SupportHardLaw().generate_pair(0)
    pair_b = SupportHardLaw().generate_pair(0)
    assert np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
    assert np.array_equal(pair_a.violate.frames, pair_b.violate.frames)


def test_seeds_differ():
    pair_a = SupportHardLaw().generate_pair(0)
    pair_b = SupportHardLaw().generate_pair(1)
    assert not np.array_equal(pair_a.obey.frames, pair_b.obey.frames)
