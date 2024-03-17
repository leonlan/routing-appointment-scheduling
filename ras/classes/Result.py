from dataclasses import dataclass

from ras.classes import Solution


@dataclass
class Result:
    solution: Solution
    runtime: float
    iterations: int
