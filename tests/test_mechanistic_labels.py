"""TDD: probes/mechanistic/labels.py -- frame-level GT label extractors."""
from physics_auditor.generator.state import BodyState, WorldState
from physics_auditor.laws import ALL_LAWS
from physics_auditor.probes.mechanistic.labels import (
    gravity_vy,
    permanence_label,
    solidity_above_shelf,
    solidity_y,
    support_label,
)


def _body(body_id, position=(0.0, 0.0), velocity=(0.0, 0.0), half_extents=None, radius=None,
          shape="box", is_static=False, mass=1.0):
    return BodyState(
        body_id=body_id, shape=shape, position=position, angle=0.0, velocity=velocity,
        angular_velocity=0.0, half_extents=half_extents, radius=radius, mass=mass,
        is_static=is_static,
    )


# --- support -----------------------------------------------------------

def test_support_label_true_when_block2_has_edge():
    state = WorldState(bodies=(_body("block0"), _body("block1"), _body("block2")),
                        support_edges=frozenset({("block2", "block1")}))
    assert support_label(state) is True


def test_support_label_false_when_block2_has_no_edge():
    state = WorldState(bodies=(_body("block0"), _body("block1"), _body("block2")),
                        support_edges=frozenset({("block1", "block0")}))
    assert support_label(state) is False


# --- permanence ----------------------------------------------------------

def test_permanence_label_true_when_ball_present():
    state = WorldState(bodies=(_body("ball"),), support_edges=frozenset())
    assert permanence_label(state) is True


def test_permanence_label_false_when_ball_absent():
    state = WorldState(bodies=(), support_edges=frozenset())
    assert permanence_label(state) is False


# --- solidity --------------------------------------------------------------

def test_solidity_y_reads_ball_position():
    state = WorldState(
        bodies=(_body("ball", position=(5.0, 42.0), shape="circle", radius=3.0),),
        support_edges=frozenset(),
    )
    assert solidity_y(state) == 42.0


def test_solidity_above_shelf_true_when_ball_higher():
    state = WorldState(
        bodies=(
            _body("shelf", position=(32.0, 20.0), half_extents=(12.0, 1.0), is_static=True),
            _body("ball", position=(32.0, 50.0), shape="circle", radius=3.0),
        ),
        support_edges=frozenset(),
    )
    assert solidity_above_shelf(state) is True


def test_solidity_above_shelf_false_when_ball_at_rest_on_shelf():
    state = WorldState(
        bodies=(
            _body("shelf", position=(32.0, 20.0), half_extents=(12.0, 1.0), is_static=True),
            _body("ball", position=(32.0, 20.5), shape="circle", radius=3.0),
        ),
        support_edges=frozenset(),
    )
    assert solidity_above_shelf(state) is False


# --- gravity -----------------------------------------------------------

def test_gravity_vy_reads_ball_velocity():
    state = WorldState(
        bodies=(_body("ball", velocity=(3.0, -17.5), shape="circle", radius=3.0),),
        support_edges=frozenset(),
    )
    assert gravity_vy(state) == -17.5


# --- real generated pairs: label balance in the post-critical window -------

def _window_states(pair):
    cf = pair.critical_frame
    return list(pair.obey.states[cf:cf + 8]) + list(pair.violate.states[cf:cf + 8])


def test_support_label_balanced_on_real_pair():
    pair = ALL_LAWS["support"]().generate_pair(0)
    labels = {support_label(s) for s in _window_states(pair)}
    assert labels == {True, False}


def test_permanence_label_balanced_on_real_pair():
    pair = ALL_LAWS["permanence"]().generate_pair(0)
    labels = {permanence_label(s) for s in _window_states(pair)}
    assert labels == {True, False}


def test_solidity_labels_balanced_and_vary_on_real_pair():
    pair = ALL_LAWS["solidity"]().generate_pair(0)
    states = _window_states(pair)
    bool_labels = {solidity_above_shelf(s) for s in states}
    assert bool_labels == {True, False}
    y_values = {solidity_y(s) for s in states}
    assert len(y_values) > 1


def test_gravity_vy_varies_on_real_pair():
    pair = ALL_LAWS["gravity"]().generate_pair(0)
    states = _window_states(pair)
    vy_values = {gravity_vy(s) for s in states}
    assert len(vy_values) > 1
