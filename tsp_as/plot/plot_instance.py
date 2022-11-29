def plot_instance(ax, params, solution=None):
    coords = params.coords

    kwargs = dict(marker="*", zorder=5, s=250, edgecolors="black")
    ax.scatter(*coords[0].T, c="tab:blue", label="depot", **kwargs)

    kwargs = dict(marker=".", zorder=3, s=300, edgecolors="black")
    ax.scatter(*coords[1:].T, c="tab:red", label="clients", **kwargs)

    if solution is not None:
        visits = [0] + solution.tour + [0]
        ax.plot(*coords[visits].T, alpha=0.75, c="tab:grey", label="tour")

    ax.set_title(params.name)
    ax.grid(color="grey", linestyle="--", linewidth=0.25)
    ax.legend(frameon=False, ncol=3)
