from __future__ import annotations

from copy import copy
from typing import Optional

from tsp_as.appointment import compute_idle_wait

from .CostEvaluator import CostEvaluator


class Solution:
    def __init__(
        self,
        data,
        tour: list[int],
        cost_evaluator: CostEvaluator,
        unassigned: Optional[list[int]] = None,
    ):
        self.data = data
        self.tour = tour  # TODO rename to visits - tour includes depot
        self.cost_evaluator = cost_evaluator
        self.schedule = None  # inter-appointment times
        self.unassigned = unassigned if unassigned is not None else []

        self._idle_times = None
        self._waiting_times = None
        self._distance = None

        self.update()

    def __deepcopy__(self, memodict):
        return Solution(
            self.data, copy(self.tour), self.cost_evaluator, copy(self.unassigned)
        )

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @property
    def cost(self):
        """
        Return the objective value. This is a weighted sum of the travel times,
        the idle times and the waiting times.
        """
        return self.cost_evaluator(self)

    @property
    def idle_times(self) -> Optional[list[float]]:
        """
        Return the idle times at each client.
        """
        return self._idle_times

    @property
    def waiting_times(self) -> Optional[list[float]]:
        """
        Return the waiting times at each client.
        """
        return self._waiting_times

    @property
    def distance(self):
        """
        Return the travel distance.
        """
        return self._distance

    def objective(self):
        """
        Alias for cost, because the ALNS interface requires ``objective()`` method.
        """
        return self.cost

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx. The insertion cost
        is the difference between the cost of the current solution and the cost of
        the candidate solution with the inserted customer.
        """
        # We create a copy of the current tour and insert the customer at the
        # specified position. Then we create a new solution object with the
        # candidate tour (which updates the cost) and compute the difference
        # in cost.
        new_tour = copy(self.tour)
        new_tour.insert(idx, customer)
        cand = Solution(self.data, new_tour, self.cost_evaluator)

        return self.cost_evaluator(cand) - self.cost_evaluator(self)

    def opt_insert(self, customer: int):
        """
        Optimally inserts the customer in the current tour.
        """
        idcs_costs = []

        for idx in range(len(self.tour) + 1):
            cost = self.insert_cost(idx, customer)
            idcs_costs.append((idx, cost))

        idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
        self.insert(idx, customer)

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)
        self.update()

    def remove(self, customer: int):
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)

    def update(self):
        """
        Update the current solution's schedule and cost.
        """
        visits = [0] + self.tour + [0]
        distance = self.data.distances[visits[1:], visits[:-1]].sum()

        schedule, idle, wait = compute_idle_wait(self.tour, self.data)

        assert len(schedule) == len(self.tour)

        self.schedule = schedule
        self._distance = distance
        self._idle_times = idle
        self._waiting_times = wait
