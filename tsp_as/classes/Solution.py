from copy import copy
from typing import Optional

import tsp_as.appointment.heavy_traffic as ht
import tsp_as.appointment.lag as lag
import tsp_as.appointment.true_optimal as to


class Solution:
    def __init__(self, data, tour: list[int], unassigned: Optional[list[int]] = None):
        self.data = data
        self.tour = tour
        self.schedule = None  # inter-appointment times
        self.unassigned = unassigned if unassigned is not None else []

        self._idle = None
        self._wait = None
        self._distance = None

        self.update()

    def __deepcopy__(self, memodict):
        return Solution(self.data, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @property
    def cost(self):
        """
        Return the objective value. This is a weighted sum of the travel times,
        the idle times and the waiting times.
        """
        return (
            self.data.omega_travel * self.distance
            + self.data.omega_idle * self.idle
            + self.data.omega_wait * self.wait
        )

    @property
    def idle(self):
        """
        Return the idle time cost.
        """
        return self._idle

    @property
    def wait(self):
        """
        Return the waiting time cost.
        """
        return self._wait

    @property
    def distance(self):
        """
        Return the travel distance.
        """
        return self._distance

    @staticmethod
    def compute_distance(tour, data):
        """
        Computes the total travel distance of the tour.
        """
        visits = [0] + tour + [0]
        return data.distances[visits[1:], visits[:-1]].sum()

    @staticmethod
    def compute_idle_wait(tour, data) -> tuple[list[float], float, float]:
        """
        Computes the idle and waiting times.

        Returns
        -------
        schedule : list[float]
            The inter-appointment times.
        idle : float
            The idle time cost.
        wait : float
            The waiting time cost.
        """
        if data.omega_idle + data.omega_wait == 0:
            # Shortcut when there are no weights for the appointment costs
            # and return the mean travel times as inter-appointment times
            pred, succ = [0] + tour[:-1], tour
            return data.distances[pred, succ], 0, 0

        if data.objective in ["htp", "hto", "htl"]:
            schedule = ht.compute_schedule(tour, data)

            if data.objective == "htp":
                # TODO Separate idle and wait costs in heavy traffic, low prio because not used
                return schedule, ht.compute_objective(tour, data)
            elif data.objective == "hto":
                idle, wait = to.compute_idle_wait(tour, schedule, data)
                return schedule, idle, wait
            elif data.objective == "htl":
                idle, wait = lag.compute_idle_wait(tour, schedule, data)
                return schedule, idle, wait

        if data.objective == "to":
            schedule, idle, wait = to.compute_schedule_and_idle_wait(tour, data)
            return schedule, idle, wait

        raise ValueError(f"{data.objective=} unknown.")

    def objective(self):
        """
        Alias for cost, because the ALNS interface requires ``objective()`` method.
        """
        return self.cost

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx. The insertion cost
        is the difference between the cost of the current solution and the cost of
        the solution with the inserted customer.
        """
        travel_cost = self.data.omega_travel * self._insert_cost_travel(idx, customer)

        idle, wait = self._insert_cost_idle_wait(idx, customer)
        idle_wait_cost = self.data.omega_idle * idle + self.data.omega_wait * wait

        return travel_cost + idle_wait_cost

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

        delta = self.data.distances[pred, cust] + self.data.distances[cust, succ]
        delta -= self.data.distances[pred, succ]

        return delta

    def _insert_cost_idle_wait(self, idx: int, customer: int) -> tuple[float, float]:
        """
        Computes the idle and wait costs for inserting customer at position idx.
        """
        # Shortcut when there are no weights for the appointment costs
        if self.data.omega_idle + self.data.omega_wait == 0:
            return 0, 0

        cand = copy(self.tour)
        cand.insert(idx, customer)
        _, new_idle, new_wait = self.compute_idle_wait(cand, self.data)

        return new_idle - self.idle, new_wait - self.wait

    def opt_insert(self, customer: int):
        """
        Optimally inserts the customer in the current tour.
        """
        idcs_costs = []

        for idx in range(len(self.tour) + 1):
            cost = self.insert_cost(idx, customer)
            idcs_costs.append((idx, cost))

        idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
        self.insert(idx, customer)

    def insert(self, idx: int, customer: int):
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)
        self.update()

    def remove(self, customer: int):
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)

    def update(self):
        """
        Update the current solution's schedule and cost.
        """
        distance = self.compute_distance(self.tour, self.data)
        schedule, idle, wait = self.compute_idle_wait(self.tour, self.data)

        assert len(schedule) == len(self.tour)

        self.schedule = schedule
        self._distance = distance
        self._idle = idle
        self._wait = wait
