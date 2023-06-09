from __future__ import annotations

from typing import Callable

import numpy as np


class CostEvaluator:
    def __init__(
        self,
        idle_wait_function: Callable,
        travel_weight: float,
        idle_weight: float,
        wait_weights: np.ndarray,
    ):
        """
        Parameters
        ----------
        idle_wait_function
            Function to be used to calculate the idle and waiting times.
        travel_weight
            Weight for travel time.
        idle_weight
            Weight for idle time.
        wait_weights
            List of waiting time weights for each client.
        """
        self.idle_wait_function = idle_wait_function
        self.travel_weight = travel_weight
        self.idle_weight = idle_weight
        self.wait_weights = wait_weights

    def __call__(self, visits, schedule, data) -> float:  # noqa: F821
        """
        Returns the cost of a solution.
        """
        distance = data.distances[[0] + visits, visits + [0]].sum()
        travel_costs = self.travel_weight * distance

        idle_times, wait_times = self.idle_wait_function(visits, schedule, data)
        idle_costs = self.idle_weight * sum(idle_times)

        wait_weights = self.wait_weights[visits]
        wait_costs = np.dot(wait_weights, wait_times)

        return travel_costs + idle_costs + wait_costs
