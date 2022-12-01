from copy import copy
from typing import Optional

import numpy as np
from alns import State

import tsp_as.appointment.heavy_traffic as ht
import tsp_as.appointment.lag as lag
import tsp_as.appointment.true_optimal as to


class Solution(State):
    tour: list[int]
    unassigned: list[int]

    def __init__(self, params, tour: list[int], unassigned: Optional[list[int]] = None):
        self.tour = tour
        self.schedule = None
        self.unassigned = unassigned if unassigned is not None else []
        self.params = params

        self._cost = None
        self.update()

    def __deepcopy__(self, memodict={}):
        self.params.trajectory.append(self)
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @property
    def cost(self):
        """
        Return the objective value. This is a weighted sum of the distance
        and the idle and waiting times. The weights are included in the cost
        computations, e.g., `compute_distance` returns the distance
        multiplied by the corresponding travel weight.
        """
        return self._cost

    @staticmethod
    def compute_distance(tour, params):
        """
        Compute the distance of the tour.
        """
        visits = [0] + tour + [0]
        return params.omega_travel * params.distances[visits[1:], visits[:-1]].sum()

    @staticmethod
    def compute_idle_wait(tour, params):
        """
        Computes the idle and waiting time cost.
        """
        # Shortcut when there are no weights for the appointment costs
        if params.omega_idle + params.omega_wait == 0:
            return None, 0

        if params.objective in ["htp", "hto", "lag"]:
            schedule = ht.compute_schedule(tour, params)

            # No need to multiply by omega here because the compute schedule
            # already takes this into account
            if params.objective == "htp":
                return schedule, ht.compute_objective(tour, params)
            elif params.objective == "hto":
                return schedule, to.compute_objective_given_schedule(
                    tour, schedule, params
                )
            elif params.objective == "lag":
                return schedule, lag.compute_objective_given_schedule(
                    tour, schedule, params
                )
        else:
            schedule, cost = to.compute_optimal_schedule(tour, params)
            return schedule, cost

    def compute_optimal_schedule(self):
        """
        Computes the optimal schedule. This function is called after the
        heuristic search to evaluate the final performance.
        """
        schedule, cost = to.compute_optimal_schedule(self.tour, self.params)
        return schedule, cost

    def objective(self):
        """
        Alias for cost, because the ALNS interface uses ``State.objective()`` method.
        """
        return self._cost

    def _insert_cost_travel(self, idx: int, cust: int) -> float:
        """
        Computes the travel cost for inserting customer at position idx.
        """
        if len(self.tour) == 0:
            pred, succ = 0, 0
        elif idx == 0:
            pred, succ = 0, self.tour[idx]
        elif idx == len(self.tour):
            pred, succ = self.tour[idx - 1], 0
        else:
            pred, succ = self.tour[idx - 1], self.tour[idx]

        cost = self.params.distances[pred, cust] + self.params.distances[cust, succ]
        cost -= self.params.distances[pred, succ]

        return self.params.omega_travel * cost

    def _insert_cost_idle_wait(self, idx: int, customer: int) -> float:
        """
        Computes the idle and wait costs for insertion customer at position idx.
        """
        # Shortcut when there are no weights for the appointment costs
        if self.params.omega_idle + self.params.omega_wait == 0:
            return 0

        cand = copy(self.tour)
        cand.insert(idx, customer)
        _, idle_wait = self.compute_idle_wait(cand, self.params)

        # No need to multiply by omega because it is already included in func
        return idle_wait

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx.
        """
        travel_cost = self._insert_cost_travel(idx, customer)
        idle_wait_cost = self._insert_cost_idle_wait(idx, customer)
        return travel_cost + idle_wait_cost

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)

    def remove(self, customer: int):
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)

    def update(self):
        """
        Update the current tour's total cost using the passed-in costs.
        """
        distance = self.compute_distance(self.tour, self.params)
        schedule, idle_wait = self.compute_idle_wait(self.tour, self.params)

        self.schedule = schedule
        self._cost = distance + idle_wait
