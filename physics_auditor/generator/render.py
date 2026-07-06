"""Rasterise a WorldState to a 64x64 RGB frame."""
import math

import numpy as np
from PIL import Image, ImageDraw

from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.generator.state import WorldState

BACKGROUND = (235, 235, 235)
GROUND_COLOR = (60, 60, 60)
BLOCK_COLORS = {
    "block0": (200, 60, 60),
    "block1": (60, 160, 60),
    "block2": (60, 60, 200),
}


def _rotated_corners(cx: float, cy: float, hx: float, hy: float, angle: float) -> list[tuple[float, float]]:
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a) for lx, ly in local]


def render(state: WorldState, size: int = 64) -> np.ndarray:
    img = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, size - GROUND_Y, size, size], fill=GROUND_COLOR)

    for body in state.bodies:
        cx, cy = body.position
        hx, hy = body.half_extents
        corners = _rotated_corners(cx, cy, hx, hy, body.angle)
        pixel_corners = [(x, size - y) for x, y in corners]
        color = BLOCK_COLORS.get(body.body_id, (128, 128, 128))
        draw.polygon(pixel_corners, fill=color)

    return np.array(img, dtype=np.uint8)
