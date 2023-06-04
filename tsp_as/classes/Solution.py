from __future__ import annotations

from copy import copy
from typing import Optional

from tsp_as.appointment import compute_idle_wait

from .CostEvaluator import CostEvaluator
from .ProblemData import ProblemData


class Solution:
    def __init__(
        self,
        data: ProblemData,
        cost_evaluator: CostEvaluator,
        visits: list[int],
        schedule: Optional[list[float]] = None,
        unassigned: Optional[list[int]] = None,
    ):
        """
        A Solution object represents a list of client visits. This may
        optionally include a schedule (i.e., inter-appointment times), but
        this is not required because it can be computed from the visits.
        """
        self.data = data
        self.visits = visits
        self.cost_evaluator = cost_evaluator
        self.schedule = schedule if schedule is not None else None
        self.unassigned = unassigned if unassigned is not None else []

        self._idle_times = None
        self._wait_times = None
        self._distance = None

        self.update()

    def __deepcopy__(self, memodict):
        return Solution(
            self.data,
            self.cost_evaluator,
            copy(self.visits),
            copy(self.schedule),
            copy(self.unassigned),
        )

    def __len__(self):
        return len(self.visits)

    def __eq__(self, other):
        return self.visits == other.visits

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
    def wait_times(self) -> Optional[list[float]]:
        """
        Return the waiting times at each client.
        """
        return self._wait_times

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
        # We create a copy of the current visits and insert the customer at the
        # specified position. Then we create a new solution object with the
        # candidate visits (which updates the cost) and compute the difference
        # in cost.
        new_visits = copy(self.visits)
        new_visits.insert(idx, customer)
        cand = Solution(self.data, self.cost_evaluator, new_visits)

        return self.cost_evaluator(cand) - self.cost_evaluator(self)

    def opt_insert(self, customer: int):
        """
        Optimally inserts the customer in the current visits.
        """
        idcs_costs = []

        for idx in range(len(self.visits) + 1):
            cost = self.insert_cost(idx, customer)
            idcs_costs.append((idx, cost))

        idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
        self.insert(idx, customer)

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.visits.insert(idx, customer)
        self.update()

    def remove(self, customer: int):
        """
        Remove the customer from the current schedule.
        """
        self.visits.remove(customer)

    def update(self):
        """
        Update the current solution's schedule and cost.
        """
        tour = [0] + self.visits + [0]
        distance = self.data.distances[tour[1:], tour[:-1]].sum()

        schedule, idle_times, wait_times = compute_idle_wait(
            self.visits, self.data, self.cost_evaluator
        )

        assert len(schedule) == len(self.visits)
        assert all(x >= 0 for x in schedule)

        self.schedule = schedule
        self._distance = distance
        self._idle_times = idle_times
        self._wait_times = wait_times
