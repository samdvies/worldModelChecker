"""Frame-level ground-truth label extractors, one per law, read straight off
a WorldState (never off pixels). Used by the decodability probe to train
linear read-outs against the *true* underlying variable a stack should be
tracking if it "understands" the law rather than shortcutting on pixels."""
from physics_auditor.generator.state import WorldState


def support_label(state: WorldState) -> bool:
    """True iff block2 (the top block of the support-law tower) currently
    has any support edge (as upper or lower member) -- i.e. it is resting on
    or being rested upon by something."""
    return any("block2" in edge for edge in state.support_edges)


def permanence_label(state: WorldState) -> bool:
    """True iff the ball body still exists in this state."""
    return any(b.body_id == "ball" for b in state.bodies)


def _ball(state: WorldState):
    return next(b for b in state.bodies if b.body_id == "ball")


def solidity_y(state: WorldState) -> float:
    """Regression target: ball's y position."""
    return _ball(state).position[1]


def solidity_above_shelf(state: WorldState) -> bool:
    """Bool target: is the ball above the shelf's top surface? Shelf geometry
    (position + half_extents) is read from the same state, so this holds
    for both obey and violate (ghosted) clips."""
    shelf = next(b for b in state.bodies if b.body_id == "shelf")
    shelf_top = shelf.position[1] + shelf.half_extents[1]
    return _ball(state).position[1] > shelf_top


def gravity_vy(state: WorldState) -> float:
    """Regression target: ball's vertical velocity."""
    return _ball(state).velocity[1]
