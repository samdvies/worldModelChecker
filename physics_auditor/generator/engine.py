"""PyMunk-backed physics engine for the support-law feasibility slice."""
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

BLOCK_IDS = ("block0", "block1", "block2")


class Engine:
    def __init__(self, config: ScenarioConfig | None = None):
        self._init_common()
        if config is not None:
            y = GROUND_Y
            for i, body_id in enumerate(BLOCK_IDS):
                width = config.widths[i]
                height = config.heights[i]
                x = config.x_centres[i]
                y_centre = y + height / 2
                self._add_block(body_id, width, height, x, y_centre)
                y += height

    def _init_common(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self._shape_body_id: dict[int, str] = {}
        self._bodies: dict[str, pymunk.Body] = {}
        self._shapes: dict[str, pymunk.Shape] = {}
        self._half_extents: dict[str, tuple[float, float]] = {}
        self._build_ground()

    def _build_ground(self) -> None:
        ground_shape = pymunk.Segment(
            self.space.static_body, (0, GROUND_Y), (WORLD_SIZE, GROUND_Y), 0.0
        )
        ground_shape.friction = FRICTION
        ground_shape.elasticity = ELASTICITY
        self.space.add(ground_shape)
        self._shape_body_id[id(ground_shape)] = "ground"

    def _add_block(
        self,
        body_id: str,
        width: float,
        height: float,
        x_centre: float,
        y_centre: float,
        angle: float = 0.0,
        velocity: tuple[float, float] = (0.0, 0.0),
        angular_velocity: float = 0.0,
        mass: float | None = None,
    ) -> None:
        if mass is None:
            mass = width * height
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = (x_centre, y_centre)
        body.angle = angle
        body.velocity = velocity
        body.angular_velocity = angular_velocity
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.friction = FRICTION
        shape.elasticity = ELASTICITY
        self.space.add(body, shape)
        self._bodies[body_id] = body
        self._shapes[body_id] = shape
        self._half_extents[body_id] = (width / 2, height / 2)
        self._shape_body_id[id(shape)] = body_id

    @classmethod
    def from_state(cls, state: WorldState) -> "Engine":
        eng = cls.__new__(cls)
        eng._init_common()
        for b in state.bodies:
            width, height = b.half_extents[0] * 2, b.half_extents[1] * 2
            eng._add_block(
                b.body_id,
                width,
                height,
                b.position[0],
                b.position[1],
                angle=b.angle,
                velocity=b.velocity,
                angular_velocity=b.angular_velocity,
                mass=b.mass,
            )
        return eng

    def ghost(self, body_id: str) -> None:
        shape = self._shapes[body_id]
        shape.filter = pymunk.ShapeFilter(categories=0xFFFFFFFF, mask=0)

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
            he = self._half_extents[body_id]
            bodies.append(
                BodyState(
                    body_id=body_id,
                    position=(body.position.x, body.position.y),
                    angle=body.angle,
                    velocity=(body.velocity.x, body.velocity.y),
                    angular_velocity=body.angular_velocity,
                    half_extents=he,
                    mass=body.mass,
                )
            )
        return WorldState(bodies=tuple(bodies), support_edges=frozenset(self._support_edges()))

    def step(self) -> WorldState:
        for _ in range(SUBSTEPS):
            self.space.step(DT)
        return self.state
