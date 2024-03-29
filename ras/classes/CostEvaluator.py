from __future__ import annotations

from typing import Callable

import numpy as np

from ras.classes.ProblemData import ProblemData


class CostEvaluator:
    def __init__(
        self,
        idle_wait_function: Callable,
        weight_travel: float,
        weight_idle: float,
        weights_wait: np.ndarray,
    ):
        """
        Parameters
        ----------
        idle_wait_function
            Function to be used to calculate the idle and waiting times.
        weight_travel
            Weight for travel time.
        weight_idle
            Weight for idle time.
        weights_wait
            List of waiting time weights for each client.
        """
        self.idle_wait_function = idle_wait_function
        self.travel_weight = weight_travel
        self.idle_weight = weight_idle
        self.wait_weights = weights_wait

    def __call__(
        self, data: ProblemData, visits: list[int], schedule: list[float]
    ) -> float:
        """
        Returns the cost of a solution.
        """
        distance = data.distances[[0] + visits, visits + [0]].sum()
        travel_costs = self.travel_weight * distance

        idle_times, wait_times = self.idle_wait_function(data, visits, schedule)
        idle_costs = self.idle_weight * sum(idle_times)

        wait_weights = self.wait_weights[visits]
        wait_costs = np.dot(wait_weights, wait_times)

        return travel_costs + idle_costs + wait_costs
