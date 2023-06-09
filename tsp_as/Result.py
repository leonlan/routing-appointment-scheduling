from dataclasses import dataclass

from tsp_as.classes import Solution


@dataclass
class Result:
    solution: Solution
    runtime: float
    iterations: int
