"""Frozen value types describing a simulated world at one instant."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyState:
    body_id: str
    position: tuple[float, float]
    angle: float
    velocity: tuple[float, float]
    angular_velocity: float
    half_extents: tuple[float, float]
    mass: float


@dataclass(frozen=True)
class WorldState:
    bodies: tuple[BodyState, ...]
    support_edges: frozenset[tuple[str, str]]
