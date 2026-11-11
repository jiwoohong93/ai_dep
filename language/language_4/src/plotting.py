"""Shared matplotlib style (reference palette: fixed categorical order, thin marks, recessive grid)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "lines.markersize": 5, "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


setup()
