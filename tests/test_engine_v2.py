"""TDD tests for the Engine v2 refactor: circles, statics, delete/gravity_off
interventions, occluders, and state_distance."""
import numpy as np

from physics_auditor.generator.config import BodySpec, ScenarioConfig
from physics_auditor.generator.engine import Engine
from physics_auditor.generator.physics import GRAVITY, GROUND_Y
from physics_auditor.generator.render import render
from physics_auditor.generator.state import BodyState, WorldState, state_distance


def _circle_spec(body_id="ball", position=(10.0, 30.0), velocity=(0.0, 0.0), is_static=False, mass=1.0, friction=0.9):
    return BodySpec(
        body_id=body_id,
        shape="circle",
        position=position,
        velocity=velocity,
        radius=3.0,
        mass=mass,
        is_static=is_static,
        friction=friction,
    )


def _box_spec(body_id="shelf", position=(32.0, 20.0), is_static=True):
    return BodySpec(
        body_id=body_id,
        shape="box",
        position=position,
        half_extents=(12.0, 1.0),
        mass=1.0,
        is_static=is_static,
    )


def test_circle_body_round_trips_through_from_state():
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(_circle_spec(velocity=(5.0, 0.0)),),
    )
    engine = Engine(config)
    state = engine.step()
    ball = next(b for b in state.bodies if b.body_id == "ball")
    assert ball.shape == "circle"
    assert ball.radius == 3.0
    assert ball.half_extents is None
    assert ball.is_static is False

    reconstructed = Engine.from_state(state)
    rstate = reconstructed.step()
    rball = next(b for b in rstate.bodies if b.body_id == "ball")
    # Stepping the original engine one further frame should match stepping
    # a from_state-reconstructed clone by the same amount.
    state_next = engine.step()
    ball_next = next(b for b in state_next.bodies if b.body_id == "ball")
    assert abs(rball.position[0] - ball_next.position[0]) < 1e-6
    assert abs(rball.position[1] - ball_next.position[1]) < 1e-6
    assert rball.shape == "circle"
    assert rball.radius == 3.0


def test_static_body_reconstructed_and_never_moves():
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(_box_spec(),),
    )
    engine = Engine(config)
    state0 = engine.state
    shelf0 = next(b for b in state0.bodies if b.body_id == "shelf")
    assert shelf0.is_static is True
    for _ in range(5):
        engine.step()
    state5 = engine.state
    shelf5 = next(b for b in state5.bodies if b.body_id == "shelf")
    assert shelf5.position == shelf0.position

    reconstructed = Engine.from_state(state5)
    rstate = reconstructed.step()
    rshelf = next(b for b in rstate.bodies if b.body_id == "shelf")
    assert rshelf.is_static is True
    assert rshelf.position == shelf5.position


def test_delete_removes_body_from_state_and_render():
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=True,
        bodies=(_circle_spec(),),
        intervention="delete",
        intervention_target="ball",
        critical_frame=2,
        n_frames=5,
    )
    engine = Engine(config)
    engine.step()
    engine.delete("ball")
    state = engine.step()
    assert all(b.body_id != "ball" for b in state.bodies)
    frame = render(state, config.occluders)
    assert frame.shape == (64, 64, 3)


def test_gravity_off_freezes_vertical_velocity():
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(_circle_spec(position=(10.0, 40.0), velocity=(5.0, 0.0)),),
    )
    engine = Engine(config)
    engine.step()
    engine.gravity_off("ball")
    vy_before = engine.state.bodies[0].velocity[1]
    for _ in range(4):
        state = engine.step()
    vy_after = state.bodies[0].velocity[1]
    assert abs(vy_after - vy_before) < 1e-6


def test_gravity_off_control_would_have_changed_vy():
    # Sanity: without the intervention, vy does change under gravity.
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(_circle_spec(position=(10.0, 40.0), velocity=(5.0, 0.0)),),
    )
    engine = Engine(config)
    engine.step()
    vy_before = engine.state.bodies[0].velocity[1]
    state = engine.step()
    vy_after = state.bodies[0].velocity[1]
    assert abs(vy_after - vy_before) > 0.01


def test_from_state_preserves_body_friction():
    """Engine.from_state must reconstruct each body with its own friction,
    not the global FRICTION constant, so oracle rollouts don't drift."""
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(
            _circle_spec(position=(10.0, 4.0 + 3.0), velocity=(30.0, 0.0), friction=0.0),
        ),
    )
    engine = Engine(config)
    state = engine.state
    rebuilt = Engine.from_state(state)
    assert rebuilt._shapes["ball"].friction == 0.0


def test_state_distance_l2_when_both_present():
    a = WorldState(
        bodies=(BodyState("b", "circle", (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, None, 3.0, 1.0, False),),
        support_edges=frozenset(),
    )
    b = WorldState(
        bodies=(BodyState("b", "circle", (3.0, 4.0), 0.0, (0.0, 0.0), 0.0, None, 3.0, 1.0, False),),
        support_edges=frozenset(),
    )
    assert state_distance(a, b) == 5.0


def test_state_distance_presence_penalty():
    a = WorldState(
        bodies=(BodyState("b", "circle", (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, None, 3.0, 1.0, False),),
        support_edges=frozenset(),
    )
    b = WorldState(bodies=(), support_edges=frozenset())
    assert state_distance(a, b) == 10.0


def test_state_distance_excludes_statics():
    a = WorldState(
        bodies=(BodyState("s", "box", (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, (1.0, 1.0), None, 1.0, True),),
        support_edges=frozenset(),
    )
    b = WorldState(
        bodies=(BodyState("s", "box", (50.0, 50.0), 0.0, (0.0, 0.0), 0.0, (1.0, 1.0), None, 1.0, True),),
        support_edges=frozenset(),
    )
    assert state_distance(a, b) == 0.0


def test_occluder_rendered_opaque_over_body_behind_it():
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=False,
        bodies=(_circle_spec(position=(32.0, 20.0)),),
        occluders=((20.0, 10.0, 24.0, 20.0),),
    )
    engine = Engine(config)
    state = engine.state
    frame_with_occluder = render(state, config.occluders)
    frame_without_occluder = render(state, ())
    assert not np.array_equal(frame_with_occluder, frame_without_occluder)
    # pixel at the ball's centre should be the occluder colour, not the ball colour
    px, py = 32, 64 - 20
    assert tuple(frame_with_occluder[py, px]) != tuple(frame_without_occluder[py, px])


def test_ghost_still_disables_collision():
    shelf = _box_spec()
    ball = _circle_spec(body_id="ball", position=(32.0, 25.0))
    config = ScenarioConfig(
        seed=0,
        law="test",
        violate=True,
        bodies=(shelf, ball),
        intervention="ghost",
        intervention_target="shelf",
        critical_frame=2,
        n_frames=20,
    )
    engine = Engine(config)
    engine.step()
    engine.ghost("shelf")
    for _ in range(15):
        state = engine.step()
    ball_final = next(b for b in state.bodies if b.body_id == "ball")
    assert ball_final.position[1] < GROUND_Y + 3.0 + 1.0  # fell through to the ground
