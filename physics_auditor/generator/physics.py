"""Shared physics/render constants (pinned)."""

GRAVITY: tuple[float, float] = (0.0, -200.0)
FPS: int = 30
SUBSTEPS: int = 4
DT: float = 1.0 / (FPS * SUBSTEPS)  # 1/120 per substep
FRICTION: float = 0.9
ELASTICITY: float = 0.0

WORLD_SIZE: int = 64
GROUND_Y: float = 4.0
