"""Rasterise a WorldState to a 64x64 RGB frame."""
import math

import numpy as np
from PIL import Image, ImageDraw

from physics_auditor.generator.physics import GROUND_Y
from physics_auditor.generator.state import WorldState

BACKGROUND = (235, 235, 235)
GROUND_COLOR = (60, 60, 60)
STATIC_COLOR = (140, 140, 140)
OCCLUDER_COLOR = (35, 35, 35)
DYNAMIC_PALETTE = [
    (200, 60, 60),
    (60, 160, 60),
    (60, 60, 200),
    (200, 160, 60),
    (160, 60, 200),
    (60, 200, 200),
]


def _rotated_corners(cx: float, cy: float, hx: float, hy: float, angle: float) -> list[tuple[float, float]]:
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a) for lx, ly in local]


def render(state: WorldState, occluders: tuple = (), size: int = 64) -> np.ndarray:
    img = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, size - GROUND_Y, size, size], fill=GROUND_COLOR)

    dyn_index = 0
    for body in state.bodies:
        if body.is_static:
            color = STATIC_COLOR
        else:
            color = DYNAMIC_PALETTE[dyn_index % len(DYNAMIC_PALETTE)]
            dyn_index += 1

        cx, cy = body.position
        if body.shape == "circle":
            r = body.radius
            bbox = [cx - r, size - (cy + r), cx + r, size - (cy - r)]
            draw.ellipse(bbox, fill=color)
        else:
            hx, hy = body.half_extents
            corners = _rotated_corners(cx, cy, hx, hy, body.angle)
            pixel_corners = [(x, size - y) for x, y in corners]
            draw.polygon(pixel_corners, fill=color)

    for x, y, w, h in occluders:
        bbox = [x, size - (y + h), x + w, size - y]
        draw.rectangle(bbox, fill=OCCLUDER_COLOR)

    return np.array(img, dtype=np.uint8)
