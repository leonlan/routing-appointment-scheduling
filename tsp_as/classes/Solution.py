from copy import copy
from typing import Optional

from alns import State

from tsp_as.classes import Params
from tsp_as.evaluations import (heavy_traffic_optimal, heavy_traffic_pure,
                                true_optimal)
from tsp_as.evaluations.tour2params import tour2params


class Solution(State):
    tour: list[int]
    unassigned: list[int]

    def __init__(
        self, params: Params, tour: list[int], unassigned: Optional[list[int]] = None
    ):
        self.tour = tour
        self.unassigned = unassigned if unassigned is not None else []
        self.params = params

    def __deepcopy__(self, memodict={}):
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @staticmethod
    def compute_objective(tour, params):
        if params.objective == "htp":
            return heavy_traffic_pure(tour, params)[1]
        elif params.objective == "hto":
            return heavy_traffic_optimal(tour, params)[1]
        else:
            return true_optimal(tour, params)[1]

    def objective(self):
        return self.compute_objective(self.tour, self.params)

    def insert_cost(self, job: int, idx: int) -> float:
        cand = copy(self.tour)
        cand.insert(idx, job)
        return self.compute_objective(cand, self.params)

    def insert(self, job: int, idx: int) -> None:
        """
        Insert the job at position idx.
        # TODO refactor to same signature as list.insert method
        """
        self.tour.insert(idx, job)

    def remove(self, job: int) -> None:
        """
        Remove the job from the current schedule.
        """
        self.tour.remove(job)