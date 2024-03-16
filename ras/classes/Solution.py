from __future__ import annotations

from ras.appointment.heavy_traffic import compute_schedule

from .CostEvaluator import CostEvaluator
from .ProblemData import ProblemData


class Route:
    """
    Represents the client visits with appointment times.

    Parameters
    ----------
    data
        The problem data.
    cost_evaluator
        The cost evaluator for the solution.
    clients
        The clients in the route.
    appointments
        The appointment times for each clients.
    """

    def __init__(
        self,
        data: ProblemData,
        cost_evaluator: CostEvaluator,
        clients: list[int],
        appointments: list[float],
    ):
        self._clients = clients
        self._appointments = appointments
        self._cost = cost_evaluator(data, clients, appointments)

    def __len__(self):
        return len(self._clients)

    def __iter__(self):
        return iter(self._clients)

    @property
    def clients(self):
        return self._clients

    @property
    def appointments(self):
        return self._appointments

    @property
    def cost(self):
        return self._cost

    @classmethod
    def from_clients(
        cls, data: ProblemData, cost_evaluator: CostEvaluator, clients: list[int]
    ):
        """
        Creates a route from the given client visits. Uses heavy traffic.

        Parameters
        ----------
        data
            The problem data.
        cost_evaluator
            The cost evaluator for the solution.
        clients
            The clients in the route.

        Returns
        -------
        Route
            The route.
        """
        appointments = compute_schedule(data, cost_evaluator, clients)
        return cls(data, cost_evaluator, clients, appointments)


class Solution:
    """
    Represents a solution to the appointment scheduling problem.

    Parameters
    ----------
    routes
        The routes of the solution.
    """

    def __init__(self, routes: list[Route]):
        self._routes = routes
        self._cost = sum([route.cost for route in routes])

    def __len__(self):
        return len([client for route in self._routes for client in route])

    @property
    def routes(self):
        return self._routes

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
        Creates a solution from a list of visits.

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
        return cls(
            [Route.from_clients(data, cost_evaluator, route) for route in routes]
        )
