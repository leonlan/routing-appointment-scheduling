from copy import copy
from typing import Optional

from alns import State

from tsp_as.classes import Params
from tsp_as.evaluations import heavy_traffic_optimal, heavy_traffic_pure, true_optimal
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

        self._distance_cost = None
        self._idle_wait_cost = None

    def __deepcopy__(self, memodict={}):
        self.params.trajectory.append(self.tour)
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @property
    def distance_cost(self):
        return self._distance_cost

    @property
    def idle_wait_cost(self):
        return self._idle_wait_cost

    @staticmethod
    def compute_distance(tour, params):
        """
        Compute the distance of the tour.
        """
        to = [0] + tour
        fr = tour + [0]
        return params.distances[to, fr].sum()

    @staticmethod
    def compute_idle_wait(tour, params):
        if params.objective == "htp":
            return heavy_traffic_pure(tour, params)[1]
        elif params.objective == "hto":
            return heavy_traffic_optimal(tour, params)[1]
        else:
            return true_optimal(tour, params)[1]

    def objective(self):
        """
        Return the objective value. This is a weighted sum of the distance
        and the idle and waiting times.
        """
        # TODO Need to add omega weights here?
        return self.distance_cost + self.idle_wait_cost
        # dist = self.compute_distance(self.tour, self.params)
        # idle_wait = self.compute_idle_wait(self.tour, self.params)

        # # TODO Need to add omega weights here?
        # return dist + idle_wait

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx.
        """
        cand = copy(self.tour)

        if len(self.tour) == 0:
            pred, succ = 0, 0
        elif idx == 0:
            pred, succ = 0, cand[idx]
        elif idx == len(self.tour):
            pred, succ = cand[idx - 1], 0
        else:
            pred, succ = cand[idx - 1], cand[idx]

        dist = self.params.distances[pred, succ]
        cand.insert(idx, customer)

        return dist + self.compute_idle_wait(cand, self.params)

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)

    def remove(self, customer: int):
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)

    def update(self, distance_cost=None, idle_wait_cost=None):
        """
        Update the current tour's total cost using the passed-in costs.

        If no costs are passed-in, then re-compute the costs.
        """
        if distance_cost is None or idle_wait_cost is None:
            self._distance_cost = self.compute_distance(self.tour, self.params)
            self._idle_wait_cost = self.compute_idle_wait(self.tour, self.params)
        else:
            self._distance_cost = distance_cost
            self._idle_wait_cost = idle_wait_cost
