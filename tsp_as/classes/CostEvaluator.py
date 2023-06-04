from __future__ import annotations

from typing import Callable

import numpy as np


class CostEvaluator:
    def __init__(
        self,
        objective_function: Callable,
        travel_weight: float,
        idle_weight: float,
        wait_weights: np.ndarray,
    ):
        """
        Parameters
        ----------
        objective_function
            Objective function to be used for evaluation.
        travel_weight
            Weight for travel time.
        idle_weight
            Weight for idle time.
        wait_weights
            List of waiting time weights for each client.
        """
        self.objective_function = objective_function
        self.travel_weight = travel_weight
        self.idle_weight = idle_weight
        self.wait_weights = wait_weights

    def __call__(self, solution: Solution) -> float:  # noqa: F821
        """
        Returns the cost of a solution.
        """
        travel_costs = self.travel_weight * solution.distance
        idle_costs = self.idle_weight * sum(solution.idle_times)

        wait_weights = self.wait_weights[solution.visits]
        wait_costs = np.dot(wait_weights, solution.wait_times)

        return travel_costs + idle_costs + wait_costs
