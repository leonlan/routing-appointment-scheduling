from copy import copy
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from alns import State

from tsp_as.classes import Params
from tsp_as.evaluations import (heavy_traffic_optimal, heavy_traffic_pure,
                                true_optimal)
from tsp_as.evaluations.tour2params import tour2params


class Solution(State):
    tour: list[int]
    unassigned: list[int]

    def __init__(
        self, params: Params, tour: list[int], unassigned: Optional[list[int]] = None
    ):
        self.tour = tour
        self.unassigned = unassigned if unassigned is not None else []
        self.params = params

    def __deepcopy__(self, memodict={}):
        self.params.trajectory.append(self.tour)
        return Solution(self.params, copy(self.tour), copy(self.unassigned))

    def __repr__(self):
        return str(self.tour)

    def __len__(self):
        return len(self.tour)

    def __eq__(self, other):
        return self.tour == other.tour

    @staticmethod
    def compute_objective(tour, params):
        if params.objective == "htp":
            return heavy_traffic_pure(tour, params)[1]
        elif params.objective == "hto":
            return heavy_traffic_optimal(tour, params)[1]
        else:
            return true_optimal(tour, params)[1]

    def objective(self):
        return self.compute_objective(self.tour, self.params)

    def insert_cost(self, idx: int, customer: int) -> float:
        """
        Compute the cost for inserting customer at position idx.
        """
        cand = copy(self.tour)
        cand.insert(idx, customer)
        return self.compute_objective(cand, self.params)

    def insert(self, idx: int, customer: int) -> None:
        """
        Insert the customer at position idx.
        """
        self.tour.insert(idx, customer)

    def remove(self, customer: int) -> None:
        """
        Remove the customer from the current schedule.
        """
        self.tour.remove(customer)

    def plot(self) -> None:
        """
        Plot the tour.
        """
        G = self.params.G
        pos = nx.spring_layout(G, seed=1)

        # Only draw nodes that are in tour
        nx.draw_networkx_nodes(G, nodelist=[0] + self.tour, pos=pos)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="whitesmoke")

        # TODO refactor this
        edges = []
        expl = [0] + self.tour + [0]
        for idx in range(len(expl) - 1):
            edges.append((expl[idx], expl[idx + 1]))

        # Plot the tour edges
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="red")

        plt.show()
