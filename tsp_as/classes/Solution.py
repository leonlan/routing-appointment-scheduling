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
        self.params.trajectory.append(self.tour)
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @staticmethod
    def compute_distance(tour, params):
        """
        Compute the distance of the tour.
        """
        # TODO double check if this is correct
        to = [0] + tour
        fr = tour + [0]
        return params.distances[to, fr].sum()

    @staticmethod
    def compute_objective(tour, params):
        if params.objective == "htp":
            return heavy_traffic_pure(tour, params)[1]
        elif params.objective == "hto":
            return heavy_traffic_optimal(tour, params)[1]
        else:
            return true_optimal(tour, params)[1]

    def objective(self):
        dist = self.compute_distance(self.tour, self.params)
        idle_wait = self.compute_objective(self.tour, self.params)

        # TODO Need to add weights here?
        return dist + idle_wait

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx.
        """
        cand = copy(self.tour)
        cand.insert(idx, customer)
        return self.compute_objective(cand, self.params)

    def insert(self, idx: int, customer: int) -> None:
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)

    def remove(self, customer: int) -> None:
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)
