"""Permanence-ext law: an EVAL-ONLY variant of Permanence that fixes the
structurally-blind-window problem of the original.

In the original Permanence law (permanence.py) the ball is deleted while
hidden behind the occluder, and the deletion only becomes *pixel-visible*
when the ball would have re-emerged -- which happens only ~9 frames before
the 48-frame clip ends. Any behavioural probe scoring surprise over a
horizon window anchored at critical_frame therefore has almost no
post-violation evidence to work with, and structurally under-detects this
violation regardless of how good the probe is.

Fix: widen the clip to n_frames=64 and re-derive the ball's speed/occluder
geometry (via direct engine simulation + search, not guesswork -- see
scripts used during development) so that:
  - the ball is still fully occluded at critical_frame (16, chosen earlier
    than the base law's 24 to leave more room), for the full seed range, and
  - the OBEY ball's leading edge clears the occluder (fully re-emerges) by
    frame ~28 at the latest across seeds 0-15, comfortably under the
    frame-34 budget, leaving >=25 frames (in fact >=36) of post-re-emergence
    evidence before the clip ends at frame 64.

Geometry (derived, not guessed): OCCLUDER=(16.0, 4.0, 10.0, 10.0) so
occ_right=26.0. Ball x0 ~ 7.0 +/- 1.0, vx ~ 26.0 +/- 1.5 (units/s). At
FPS=30 that's displacement/frame = vx/30 ~ 0.83-0.92 units/frame. At
critical_frame=16 the ball's centre lands in [~19.6, ~22.2], comfortably
inside the occluder with >=0.6 unit margin on both edges for every seed in
0-15 (see test_full_occlusion_at_critical_all_seeds-style check in the
validity tests below). The ball clears occ_right+RADIUS by frame <=28 for
every one of those seeds (max observed 28, min 24).

EVAL-ONLY invariant: this law is registered in ALL_LAWS but intentionally
NOT part of TRAIN_LAWS (see physics_auditor/laws/__init__.py). It must
never contribute frames to the predictor/encoder training distribution
(physics_auditor/generator/dataset.py train_clips()/val_clips()). See that
module's docstring/registry split for the mechanism; dataset.py itself is
owned by another workstream and is not modified here.
"""
from dataclasses import replace

import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.generator.simulate import simulate
from physics_auditor.laws.base import MinimalPair

RADIUS = 3.0
# Derived (see module docstring) so the ball is fully hidden at
# critical_frame=16 and fully re-emerges by frame ~28, across seeds 0-15.
OCCLUDER = (16.0, 4.0, 10.0, 10.0)  # (x, y, w, h)
N_FRAMES = 64
CRITICAL_FRAME = 16
X0_BASE, X0_JITTER = 7.0, 1.0
VX_BASE, VX_JITTER = 26.0, 1.5


class PermanenceExtLaw:
    name = "permanence-ext"

    def generate_pair(self, seed: int) -> MinimalPair:
        rng = np.random.default_rng(seed)
        x0 = float(X0_BASE + rng.uniform(-X0_JITTER, X0_JITTER))
        vx = float(VX_BASE + rng.uniform(-VX_JITTER, VX_JITTER))
        y0 = GROUND_Y + RADIUS

        ball = BodySpec(
            body_id="ball",
            shape="circle",
            position=(x0, y0),
            velocity=(vx, 0.0),
            radius=RADIUS,
            mass=1.0,
            friction=0.0,
        )

        obey_config = ScenarioConfig(
            seed=seed,
            law=self.name,
            violate=False,
            bodies=(ball,),
            n_frames=N_FRAMES,
            critical_frame=CRITICAL_FRAME,
            intervention="delete",
            intervention_target="ball",
            occluders=(OCCLUDER,),
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
