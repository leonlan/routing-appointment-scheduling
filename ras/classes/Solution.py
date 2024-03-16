from __future__ import annotations

from ras.appointment.heavy_traffic import compute_schedule

from .CostEvaluator import CostEvaluator
from .ProblemData import ProblemData


class Solution:
    """
    Represents a solution to the appointment scheduling problem.

    Parameters
    ----------
    data
        The problem data.
    cost_evaluator
        The cost evaluator (objective function) for the solution.
    routes
        The routes of the solution.
    schedule
        The schedule of the solution.
    """

    def __init__(
        self,
        data: ProblemData,
        cost_evaluator: CostEvaluator,
        routes: list[list[int]],
        schedule: list[int],
    ):
        self._routes = routes
        self._schedule = schedule
        self._cost = sum(
            [
                cost_evaluator(data, route, appointments)
                for route, appointments in zip(routes, schedule)
            ]
        )

    def __len__(self):
        return len([client for route in self._routes for client in route])

    @property
    def routes(self):
        return self._routes

    @property
    def schedule(self):
        return self._schedule

    @property
    def cost(self):
        return self._cost

    def objective(self):
        return self._cost

    @classmethod
    def from_routes(
        cls, data: ProblemData, cost_evaluator: CostEvaluator, routes: list[list[int]]
    ):
        """
        Creates a solution from the given routes.

        Parameters
        ----------
        data
            The problem data.
        cost_evaluator
            The cost evaluator for the solution.
        routes
            The routes of the solution.

        Returns
        -------
        Solution
            The solution.
        """
        schedule = [compute_schedule(data, cost_evaluator, route) for route in routes]
        return cls(data, cost_evaluator, routes, schedule)
