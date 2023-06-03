from __future__ import annotations

import numpy as np


class CostEvaluator:
    def __init__(
        self, travel_penalty: float, idle_penalty: float, wait_penalties: list[float]
    ):
        """
        Parameters
        ----------
        travel_penalty
            Penalty for travel time.
        idle_penalty
            Penalty for idle time.
        wait_penalties
            List of wait penalties for each client.
        """
        self.travel_penalty = travel_penalty
        self.idle_penalty = idle_penalty
        self.wait_penalties = np.array(wait_penalties)

    def __call__(self, solution: Solution) -> float:  # noqa: F821
        """
        Returns the cost of a solution.
        """
        travel_costs = self.travel_penalty * solution.distance
        idle_costs = np.dot(self.idle_penalty, solution.idle_times).sum()

        wait_penalties = self.wait_penalties[solution.tour]
        wait_costs = np.dot(wait_penalties, solution.waiting_times)

        return travel_costs + idle_costs + wait_costs
