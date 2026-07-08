"""Support-hard law: an EVAL-ONLY variant of Support designed to defeat the
pixel-static baseline's free ride.

The original Support law (support.py) is pixel-trivial: the only motion in
either clip is the tower's own collapse, which happens ONLY in the violate
clip. A pixel-difference/pixel-static baseline therefore gets AUROC 1.0 for
free -- it never has to understand *support*, it just has to notice "did
anything move". This tells us nothing about whether a probe understands the
physics.

Design (pixel decoy): keep the SAME law-relevant tower (Tower A: 3 stacked
blocks "block0/1/2", ghost-intervention on "block1", identical to
support.py) but add a second, unrelated tower (Tower B: "decoyA"/"decoyB")
that is DELIBERATELY unstable and collapses under pure gravity -- with no
ghost/delete/any intervention -- in BOTH the obey and violate clip. Because
Tower B's dynamics don't depend on config.violate at all, it renders
byte-identically in obey and violate (same seed => same rng draws => same
initial state => same deterministic physics), so the minimal-pair discipline
(prefix-identical, single-variable-diff) is untouched. But its motion
"swamps" naive frame-difference statistics: a pixel-static probe now sees
motion in BOTH classes, and the amount/timing of that motion is randomised
per-seed, so its magnitude sometimes lands squarely in the post-critical-
frame scoring window (large "obey" surprise) and sometimes has already
finished or not yet started (near-zero "obey" surprise) -- while the
lawful/oracle prediction of Tower B's motion is trivial to get right in
BOTH clips (it's just ordinary unforced gravity), so the *oracle* is
untouched (AUROC 1.0): only pixel-level baselines are degraded.

Mechanism: Tower B is a 2-block stack (decoyA static-ish base, decoyB on
top, offset by a fixed horizontal `overhang` that alone is NOT enough to
tip it at rest). A per-seed `gap` (vertical air-gap under decoyB, free-fall
distance before it lands on decoyA) supplies the *extra* momentum needed to
tip it over; `gap` is drawn from a wide per-seed range so the tip -- and
hence the post-critical-frame visible motion -- lands at wildly different
points in the timeline for different seeds (sometimes it has already
settled by critical_frame, sometimes it's mid-collapse squarely in the
scoring window, sometimes it hasn't been disturbed at all because the fall
was too short to destabilise it). This is a smooth, non-chaotic knob (it's
governed by ordinary free-fall kinematics, not a bifurcation/threshold
sensitivity), so it doesn't introduce cross-run/seed flakiness the way
tuning `overhang` right at its tipping point would.

Geometry / world-bounds note (see generator/physics.py: WORLD_SIZE=64,
GROUND_Y=4.0, and the historical interpenetration/off-the-edge-of-the-world
bug this codebase has hit before): Tower A's random geometry (support.py)
can occupy x roughly in [20, 44] in the worst case. Tower B is pinned with
DECOY_X=9.0, `overhang` fixed at a SAFE (non-threshold) value of 4.3, and
`gap` capped at 90.0 -- empirically verified (see development notes) to
keep every body's rotated bounding box within [1, 63] and >=1 unit clear of
Tower A's leftmost extent, for eval seeds 0-15. A LARGER gap (~100+) or
`overhang` closer to Tower B's own at-rest tipping threshold (~6.7 for this
geometry) measurably lowers the pixel floor further but was rejected here
because it occasionally drives decoyB either through the ground segment's
finite x-extent (falls forever once x<0 or x>64 -- there is no side wall)
or into Tower A -- i.e. exactly the class of bug this module's design
deliberately avoids. The literal "placed >=25 units away" framing from the
design brief is not achievable as a HARD invariant in a 64-unit-wide world
with a 3-block tower that can itself occupy the middle third: what's
actually enforced (and load-bearing) is non-overlap with >=1 unit clearance
in all directions, which is what test (c)/(d) below rely on.

EVAL-ONLY invariant: this law is registered in ALL_LAWS but intentionally
NOT part of TRAIN_LAWS (see physics_auditor/laws/__init__.py docstring/
registry split). dataset.py (owned by another workstream) is not modified
here; see this repo's task notes for the one-line follow-up required there.
"""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair

WORLD_CENTRE_X = 32.0

# Tower B (decoy) geometry -- see module docstring for how these were derived
# and why they're safe. All fixed except `gap`, which is the sole per-seed
# random draw controlling decoy timing/magnitude.
DECOY_X = 9.0
DECOY_W0, DECOY_H0 = 14.0, 8.0  # decoyA (base)
DECOY_W1, DECOY_H1 = 13.0, 8.0  # decoyB (falls/topples)
DECOY_OVERHANG = 4.3
DECOY_GAP_LO, DECOY_GAP_HI = 0.0, 90.0


class SupportHardLaw:
    name = "support-hard"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        widths = tuple(float(w) for w in rng.uniform(8, 14, size=3))
        heights = tuple(float(h) for h in rng.uniform(5, 8, size=3))
        tower_offset = float(rng.uniform(-4, 4))
        jitters = rng.uniform(-1, 1, size=3)
        x_centres = tuple(float(WORLD_CENTRE_X + tower_offset + j) for j in jitters)

        bodies = []
        y = GROUND_Y
        for i in range(3):
            width, height = widths[i], heights[i]
            y_centre = y + height / 2
            bodies.append(
                BodySpec(
                    body_id=f"block{i}",
                    shape="box",
                    position=(x_centres[i], y_centre),
                    mass=width * height,
                    half_extents=(width / 2, height / 2),
                )
            )
            y += height

        # Tower B (decoy): same rng stream, drawn AFTER Tower A so that
        # Tower A's own geometry is byte-identical to plain SupportLaw for
        # the same seed. Its dynamics never read config.violate, so it is
        # identical between obey and violate by construction.
        gap = float(rng.uniform(DECOY_GAP_LO, DECOY_GAP_HI))
        decoy_a = BodySpec(
            body_id="decoyA",
            shape="box",
            position=(DECOY_X, GROUND_Y + DECOY_H0 / 2),
            mass=DECOY_W0 * DECOY_H0,
            half_extents=(DECOY_W0 / 2, DECOY_H0 / 2),
        )
        decoy_b_y = GROUND_Y + DECOY_H0 + gap + DECOY_H1 / 2
        decoy_b = BodySpec(
            body_id="decoyB",
            shape="box",
            position=(DECOY_X + DECOY_OVERHANG, decoy_b_y),
            mass=DECOY_W1 * DECOY_H1,
            half_extents=(DECOY_W1 / 2, DECOY_H1 / 2),
        )
        bodies += [decoy_a, decoy_b]

        obey_config = ScenarioConfig(
            seed=seed,
            law=self.name,
            violate=False,
            bodies=tuple(bodies),
            intervention="ghost",
            intervention_target="block1",
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
