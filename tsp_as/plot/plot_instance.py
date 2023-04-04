def plot_instance(ax, data, solution=None):
    coords = data.coords

    kwargs = {"marker": "*", "zorder": 5, "s": 250, "edgecolors": "black"}
    ax.scatter(*coords[0].T, c="tab:blue", label="depot", **kwargs)

    kwargs = {"marker": ".", "zorder": 3, "s": 300, "edgecolors": "black"}
    ax.scatter(*coords[1:].T, c="tab:red", label="clients", **kwargs)

    if solution is not None:
        visits = [0] + solution.tour + [0]
        ax.plot(*coords[visits].T, alpha=0.75, c="tab:grey", label="tour")

    ax.set_title(data.name)
    ax.grid(color="grey", linestyle="--", linewidth=0.25)
    ax.legend(frameon=True, ncol=3)
