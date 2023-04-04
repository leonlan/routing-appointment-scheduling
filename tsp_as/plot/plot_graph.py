import networkx as nx


def plot_graph(ax, data, solution=None):
    """
    Plots the instance on a graph. This is an alternative to `plot_instance`,
    because it much easier to add edge labels to graph plots.
    """
    coords = data.coords

    G = nx.DiGraph()

    visits = [0] + solution.tour + [0]
    pos = dict(enumerate(coords))

    labels = {}
    for idx in range(len(visits) - 1):
        to, fr = visits[idx], visits[idx + 1]
        edge = (to, fr)
        G.add_edge(*edge)

        interarrival_time = int(solution.schedule[idx])
        dist = data.distances[edge]
        dist_scv = data.distances_scv[edge]

        label = f"x={interarrival_time}\n T=({dist:.0f}, {dist_scv:.2f})"
        labels[edge] = label

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=["r"] + ["#1f78b4"] * (data.dimension - 1),
        node_size=150,
        edge_color="black",
    )

    _ = nx.draw_networkx_labels(
        G,
        pos={k: v * [1, 1.01] for k, v in pos.items()},
        ax=ax,
        font_size=8,
        labels={
            k: f"B=({data.service[k]:.0f}, {data.service_scv[k]:.2f})"
            for k in pos.keys()
        },
    )

    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        ax=ax,
        font_size=9,
        edge_labels=labels,
        font_color="red",
    )

    title = f"Instance: {data.name}\n Cost: {solution.cost:.2f}"
    ax.set_title(title)
    ax.grid(color="grey", linestyle="--", linewidth=0.25)
