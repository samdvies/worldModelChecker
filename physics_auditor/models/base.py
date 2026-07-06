"""Common protocol for model adapters under test."""
from typing import Protocol

from physics_auditor.generator.clip import Clip


class ModelAdapter(Protocol):
    name: str

    def surprise(self, clip: Clip, critical_frame: int, horizon: int = 8) -> float: ...
