"""Shared figure style: one colour per model everywhere, recessive axes, thin marks."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# categorical palette, fixed order: a model keeps its colour in every figure
MODEL_COLORS = {
    "Q": "#2a78d6",      # blue
    "QK": "#eb6834",     # orange
    "HOP": "#1baf7a",    # aqua
    "HYB": "#eda100",    # yellow
    "HOPC": "#e87ba4",   # magenta
    "HOPK": "#008300",   # green
    "HYBK": "#6250d6",   # violet
    "STIM": "#898781",   # neutral grey (reference model)
}
DATA_COLOR = "#0b0b0b"   # observed behaviour is always drawn in ink
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#d03b3b"       # reference lines (theory)

# sequential ramp (one hue, light -> dark) for ordered parameter values
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#5598e7", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def seq_colors(n):
    """n ordered colours from the sequential ramp."""
    if n == 1:
        return [SEQ_BLUE[3]]
    return [SEQ_CMAP(0.25 + 0.75 * i / (n - 1)) for i in range(n)]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "pdf.fonttype": 42,
    })


def panel_label(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right", color=INK)


def save(fig, path_without_ext):
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    fig.savefig(path_without_ext + ".pdf")
    fig.savefig(path_without_ext + ".png", dpi=200)
    plt.close(fig)
