from __future__ import annotations

from copy import copy
from typing import Optional

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule

from .CostEvaluator import CostEvaluator
from .ProblemData import ProblemData


class Solution:
    def __init__(
        self,
        data: ProblemData,
        cost_evaluator: CostEvaluator,
        visits: list[int],
        schedule: list[float],
        unassigned: Optional[list[int]] = None,
    ):
        """
        A Solution object represents a list of client visits. This may
        optionally include a schedule (i.e., inter-appointment times), but
        this is not required because it can be computed from the visits.
        """
        self.data = data
        self.cost_evaluator = cost_evaluator
        self.visits = visits
        self.schedule = schedule
        self.unassigned = unassigned if unassigned is not None else []

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
        Return the objective value. This is a weighted sum of the travel times,
        the idle times and the waiting times.
        """
        return self.cost_evaluator(self)

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.visits.insert(idx, customer)
        self.schedule = compute_ht_schedule(self.visits, self.data, self.cost_evaluator)
        self.update()

    def update(self):
        """
        Update the current solution's schedule and cost.
        """
        tour = [0] + self.visits + [0]
        distance = self.data.distances[tour[1:], tour[:-1]].sum()

        idle_times, wait_times = self.cost_evaluator.idle_wait_function(
            self.visits,
            self.schedule,
            self.data,
        )

        assert len(self.schedule) == len(self.visits)
        assert all(x >= 0 for x in self.schedule)

        self._distance = distance
        self._idle_times = idle_times
        self._wait_times = wait_times
