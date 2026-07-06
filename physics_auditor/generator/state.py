"""Frozen value types describing a simulated world at one instant."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyState:
    body_id: str
    shape: str  # 'box' | 'circle'
    position: tuple[float, float]
    angle: float
    velocity: tuple[float, float]
    angular_velocity: float
    half_extents: tuple[float, float] | None  # boxes
    radius: float | None  # circles
    mass: float
    is_static: bool
    friction: float = 0.9


@dataclass(frozen=True)
class WorldState:
    bodies: tuple[BodyState, ...]
    support_edges: frozenset[tuple[str, str]]


def state_distance(a: WorldState, b: WorldState) -> float:
    """L2 position error over the union of dynamic body_ids.

    A body present in one state but missing from the other incurs a fixed
    penalty of 10.0. Static bodies are excluded entirely.
    """
    a_by_id = {bd.body_id: bd for bd in a.bodies if not bd.is_static}
    b_by_id = {bd.body_id: bd for bd in b.bodies if not bd.is_static}
    total = 0.0
    for body_id in set(a_by_id) | set(b_by_id):
        if body_id in a_by_id and body_id in b_by_id:
            pa = a_by_id[body_id].position
            pb = b_by_id[body_id].position
            total += ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
        else:
            total += 10.0
    return total
