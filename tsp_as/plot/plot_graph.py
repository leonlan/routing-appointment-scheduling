import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(ax, params, solution=None):
    """
    Plots the instance on a graph. This is an alternative to `plot_instance`,
    because it is easy to add labels to graph plots.
    """
    coords = params.coords

    G = nx.DiGraph()

    visits = [0] + solution.tour + [0]
    pos = dict(enumerate(coords))

    appointment_times = {}
    for idx in range(len(visits) - 1):
        to, fr = visits[idx], visits[idx + 1]
        G.add_edge(to, fr)

        e = (to, fr)
        appointment_times[
            (to, fr)
        ] = f"x = {int(solution.schedule[idx])},\
\n(T:{params.distances[e]:.0f}, {params.distances_scv[e]:.2f})"

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=["r"] + ["#1f78b4"] * (params.dimension - 1),
        node_size=150,
        edge_color="black",
    )

    nx.draw_networkx_labels(
        G,
        pos={k: v * [1, 1.01] for k, v in pos.items()},
        ax=ax,
        labels={
            k: f"B:({params.service[k]:.0f}, {params.service_scv[k]:.2f})"
            for k in pos.keys()
        },
    )

    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        ax=ax,
        edge_labels=appointment_times,
        font_color="red",
    )

    ax.set_title(params.name)
    ax.grid(color="grey", linestyle="--", linewidth=0.25)
