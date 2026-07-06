"""Common types shared by all physical laws."""
from dataclasses import dataclass
from typing import Protocol

from physics_auditor.generator.clip import Clip


@dataclass
class MinimalPair:
    obey: Clip
    violate: Clip
    critical_frame: int
    differing_variable: str = "violate"


class Law(Protocol):
    name: str

    def generate_pair(self, seed: int) -> MinimalPair: ...
