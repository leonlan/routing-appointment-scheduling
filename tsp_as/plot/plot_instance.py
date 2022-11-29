def plot_instance(ax, params, solution=None):
    coords = params.coords
    depot_coords = coords[0].T

    kwargs = dict(marker="*", zorder=5, s=250, edgecolors="black")
    ax.scatter(*depot_coords, c="tab:blue", label="depot", **kwargs)

    kwargs = dict(marker=".", zorder=3, s=300, edgecolors="black")
    ax.scatter(*coords[1:].T, c="tab:red", label="clients", **kwargs)

    if solution is not None:
        visits = coords[[0] + solution.tour + [0]].T
        ax.plot(*visits, alpha=0.75, c="tab:grey", label="tour")

    # TODO plot labels with inter appointment times?

    ax.set_title(params.name)
    ax.grid(color="grey", linestyle="--", linewidth=0.25)
    ax.legend(frameon=False, ncol=3)
