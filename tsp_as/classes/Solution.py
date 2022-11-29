from copy import copy
from typing import Optional

from alns import State

import tsp_as.appointment.heavy_traffic as ht
import tsp_as.appointment.true_optimal as to


class Solution(State):
    tour: list[int]
    unassigned: list[int]

    def __init__(self, params, tour: list[int], unassigned: Optional[list[int]] = None):
        self.tour = tour
        self.schedule = None
        self.unassigned = unassigned if unassigned is not None else []
        self.params = params

        self._cost = None
        self.update()

    def __deepcopy__(self, memodict={}):
        self.params.trajectory.append(self)
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @property
    def cost(self):
        """
        Return the objective value. This is a weighted sum of the distance
        and the idle and waiting times.
        """
        return self._cost

    @staticmethod
    def compute_distance(tour, params):
        """
        Compute the distance of the tour.
        """
        visits = [0] + tour + [0]
        return params.distances[visits[1:], visits[:-1]].sum()

    @staticmethod
    def compute_idle_wait(tour, params):
        if params.objective in ["htp", "hto"]:
            schedule = ht.compute_schedule(tour, params)

            if params.objective == "htp":
                return schedule, ht.compute_objective(tour, params)
            elif params.objective == "hto":
                return schedule, to.compute_objective(tour, params)
        else:
            schedule, cost = to.true_optimal(tour, params)
            return schedule, cost

    def objective(self):
        """
        Alias for cost, because the ALNS interface uses ``State.objective()`` method.
        """
        return self._cost

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

        _, idle_wait = self.compute_idle_wait(cand, self.params)
        return dist + idle_wait

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

    def update(self):
        """
        Update the current tour's total cost using the passed-in costs.
        """
        distance = self.compute_distance(self.tour, self.params)
        schedule, idle_wait = self.compute_idle_wait(self.tour, self.params)

        # TODO Need to add omega weights here? Or should it be included in
        # the computation of the costs?
        self.schedule = schedule
        self._cost = distance + idle_wait
