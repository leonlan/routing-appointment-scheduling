from __future__ import annotations

from typing import Optional

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
        self.visits = visits
        self.schedule = schedule
        self.unassigned = unassigned if unassigned is not None else []

        self.cost = cost_evaluator(visits, schedule, data)

    def __len__(self):
        return len(self.visits)

    def objective(self):
        """
        Return the objective value. This is a weighted sum of the travel times,
        the idle times and the waiting times.
        """
        return self.cost
