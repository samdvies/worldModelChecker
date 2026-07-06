"""PyMunk-backed physics engine (v2: circles, statics, delete/gravity_off)."""
import pymunk

from physics_auditor.generator.config import ScenarioConfig
from physics_auditor.generator.physics import (
    DT,
    ELASTICITY,
    FRICTION,
    GRAVITY,
    GROUND_Y,
    SUBSTEPS,
    WORLD_SIZE,
)
from physics_auditor.generator.state import BodyState, WorldState


def _zero_gravity_velocity_func(body, gravity, damping, dt):
    """velocity_func override: integrate velocity with no gravity, no damping."""
    body.update_velocity(body, (0.0, 0.0), 1.0, dt)


class Engine:
    def __init__(self, config: ScenarioConfig | None = None):
        self._init_common()
        if config is not None:
            for spec in config.bodies:
                self._add(
                    body_id=spec.body_id,
                    shape_kind=spec.shape,
                    position=spec.position,
                    velocity=spec.velocity,
                    angle=0.0,
                    angular_velocity=0.0,
                    half_extents=spec.half_extents,
                    radius=spec.radius,
                    mass=spec.mass,
                    friction=spec.friction,
                    is_static=spec.is_static,
                )

    def _init_common(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self._shape_body_id: dict[int, str] = {}
        self._bodies: dict[str, pymunk.Body] = {}
        self._shapes: dict[str, pymunk.Shape] = {}
        self._shape_kind: dict[str, str] = {}
        self._half_extents: dict[str, tuple[float, float] | None] = {}
        self._radius: dict[str, float | None] = {}
        self._mass: dict[str, float] = {}
        self._is_static: dict[str, bool] = {}
        self._friction: dict[str, float] = {}
        self._build_ground()

    def _build_ground(self) -> None:
        ground_shape = pymunk.Segment(
            self.space.static_body, (0, GROUND_Y), (WORLD_SIZE, GROUND_Y), 0.0
        )
        ground_shape.friction = FRICTION
        ground_shape.elasticity = ELASTICITY
        self.space.add(ground_shape)
        self._shape_body_id[id(ground_shape)] = "ground"

    def _add(
        self,
        body_id: str,
        shape_kind: str,
        position: tuple[float, float],
        velocity: tuple[float, float],
        angle: float,
        angular_velocity: float,
        half_extents: tuple[float, float] | None,
        radius: float | None,
        mass: float,
        friction: float,
        is_static: bool,
    ) -> None:
        if is_static:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            if shape_kind == "circle":
                moment = float("inf")  # no rotational dynamics for balls in this world
            else:
                moment = pymunk.moment_for_box(mass, (half_extents[0] * 2, half_extents[1] * 2))
            body = pymunk.Body(mass, moment)
        body.position = position
        body.angle = angle
        if not is_static:
            body.velocity = velocity
            body.angular_velocity = angular_velocity

        if shape_kind == "circle":
            shape = pymunk.Circle(body, radius)
        else:
            shape = pymunk.Poly.create_box(body, (half_extents[0] * 2, half_extents[1] * 2))
        shape.friction = friction
        shape.elasticity = ELASTICITY

        self.space.add(body, shape)
        self._bodies[body_id] = body
        self._shapes[body_id] = shape
        self._shape_kind[body_id] = shape_kind
        self._half_extents[body_id] = half_extents
        self._radius[body_id] = radius
        self._mass[body_id] = mass
        self._is_static[body_id] = is_static
        self._friction[body_id] = friction
        self._shape_body_id[id(shape)] = body_id

    @classmethod
    def from_state(cls, state: WorldState) -> "Engine":
        """Reconstruct a fully lawful world (normal gravity, all solid) from a state."""
        eng = cls.__new__(cls)
        eng._init_common()
        for b in state.bodies:
            eng._add(
                body_id=b.body_id,
                shape_kind=b.shape,
                position=b.position,
                velocity=b.velocity,
                angle=b.angle,
                angular_velocity=b.angular_velocity,
                half_extents=b.half_extents,
                radius=b.radius,
                mass=b.mass,
                friction=b.friction,
                is_static=b.is_static,
            )
        return eng

    def ghost(self, body_id: str) -> None:
        shape = self._shapes[body_id]
        shape.filter = pymunk.ShapeFilter(categories=0xFFFFFFFF, mask=0)

    def delete(self, body_id: str) -> None:
        body = self._bodies.pop(body_id)
        shape = self._shapes.pop(body_id)
        self.space.remove(body, shape)
        self._shape_body_id.pop(id(shape), None)
        self._shape_kind.pop(body_id, None)
        self._half_extents.pop(body_id, None)
        self._radius.pop(body_id, None)
        self._mass.pop(body_id, None)
        self._is_static.pop(body_id, None)
        self._friction.pop(body_id, None)

    def gravity_off(self, body_id: str) -> None:
        self._bodies[body_id].velocity_func = _zero_gravity_velocity_func

    def _support_edges(self) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()

        def make_cb():
            def cb(arbiter):
                shapes = arbiter.shapes
                ids = [self._shape_body_id.get(id(s)) for s in shapes]
                if None in ids:
                    return
                normal = arbiter.contact_point_set.normal
                if abs(normal.y) <= abs(normal.x):
                    return
                a_id, b_id = ids
                a_y = GROUND_Y if a_id == "ground" else self._bodies[a_id].position.y
                b_y = GROUND_Y if b_id == "ground" else self._bodies[b_id].position.y
                if a_y >= b_y:
                    upper, lower = a_id, b_id
                else:
                    upper, lower = b_id, a_id
                if upper == "ground":
                    return
                edges.add((upper, lower))

            return cb

        for body in self._bodies.values():
            body.each_arbiter(make_cb())
        return edges

    @property
    def state(self) -> WorldState:
        bodies = []
        for body_id, body in self._bodies.items():
            is_static = self._is_static[body_id]
            velocity = (0.0, 0.0) if is_static else (body.velocity.x, body.velocity.y)
            angular_velocity = 0.0 if is_static else body.angular_velocity
            bodies.append(
                BodyState(
                    body_id=body_id,
                    shape=self._shape_kind[body_id],
                    position=(body.position.x, body.position.y),
                    angle=body.angle,
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    half_extents=self._half_extents[body_id],
                    radius=self._radius[body_id],
                    mass=self._mass[body_id],
                    is_static=is_static,
                    friction=self._friction[body_id],
                )
            )
        return WorldState(bodies=tuple(bodies), support_edges=frozenset(self._support_edges()))

    def step(self) -> WorldState:
        for _ in range(SUBSTEPS):
            self.space.step(DT)
        return self.state
