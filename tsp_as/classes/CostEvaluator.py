from __future__ import annotations

import numpy as np


class CostEvaluator:
    def __init__(
        self, travel_weight: float, idle_weight: float, wait_weights: list[float]
    ):
        """
        Parameters
        ----------
        travel_weight
            Weight for travel time.
        idle_weight
            Weight for idle time.
        wait_weights
            List of wait weights for each client.
        """
        self.travel_weight = travel_weight
        self.idle_weight = idle_weight
        self.wait_weights = np.array(wait_weights)

    def __call__(self, solution: Solution) -> float:  # noqa: F821
        """
        Returns the cost of a solution.
        """
        travel_costs = self.travel_weight * solution.distance
        idle_costs = np.dot(self.idle_weight, solution.idle_times).sum()

        wait_weights = self.wait_weights[solution.tour]
        wait_costs = np.dot(wait_weights, solution.waiting_times)

        return travel_costs + idle_costs + wait_costs
