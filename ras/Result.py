from dataclasses import dataclass

from .Solution import Solution


@dataclass
class Result:
    solution: Solution
    runtime: float
    iterations: int
