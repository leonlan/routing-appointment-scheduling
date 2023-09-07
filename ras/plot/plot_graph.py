import networkx as nx


def plot_graph(ax, data, solution=None):
    """
    Plots the instance on a graph. This is an alternative to `plot_instance`,
    because it much easier to add edge labels to graph plots.
    """
    coords = data.coords

    G = nx.DiGraph()

    tour = [0] + solution.visits + [0]
    pos = dict(enumerate(coords))

    labels = {}
    for idx in range(len(tour) - 1):
        to, fr = tour[idx], tour[idx + 1]
        edge = (to, fr)
        G.add_edge(*edge)

        if idx < len(tour) - 2:  # not the last edge
            interarrival_time = int(solution.schedule[idx])
            # label = f"x={interarrival_time},\n E[T]={dist:.0f})"
            label = f"x={interarrival_time}"
            labels[edge] = label

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=["r"] + ["#1f78b4"] * (data.dimension - 1),
        node_size=200,
        edge_color="black",
    )

    # offsets = [
    #     [0, -0.25],
    #     [0, -0.25],
    #     [0, -0.25],
    #     [0, -0.25],
    #     [0, 0.2],
    #     [0, 0.2],
    # ]
    offset = [0, 0.1]
    _ = nx.draw_networkx_labels(
        G,
        pos={k: v + offset for idx, (k, v) in enumerate(pos.items())},
        ax=ax,
        font_size=24,
        labels={
            # k: f"B=({data.service[k]:.0f}, {data.service_scv[k]:.2f})"
            # k: f"E[B]={data.service[k]:.0f},\n $c^2_B$={data.service_scv[k]:.2f}"
            k: f"$c^2_B$={data.service_scv[k]:.2f}"
            for k in pos.keys()
        },
    )

    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        ax=ax,
        font_size=24,
        edge_labels=labels,
        font_color="red",
    )

    # ax.set_title(title)
    # ax.grid(color="grey", linestyle="--", linewidth=0.15)
