"""Generate striking 3D element embedding maps for presentations.

Creates a series of high-quality 3D scatter plots showing elements
positioned in reduced embedding space, colored by element group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from elementembeddings.core import Embedding
from elementembeddings.utils.config import ELEMENT_GROUPS_PALETTES

OUTPUT_DIR = Path(__file__).parent / "3d_maps"
OUTPUT_DIR.mkdir(exist_ok=True)

# Curated embedding schemes that produce visually interesting 3D maps
EMBEDDINGS = {
    "magpie": "Magpie",
    "mat2vec": "Mat2Vec",
    "matscholar": "Matscholar",
    "megnet16": "MEGNet",
    "skipatom": "SkipAtom",
    "oliynyk": "Oliynyk",
    "xenonpy": "XenonPy",
    "cgnf": "CGNF",
}

REDUCERS = ["umap", "pca", "tsne"]

# Dark theme colors
BG_COLOR = "#0a0a1a"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#1a1a3a"

# Refined palette with more saturated, vibrant colors for dark background
GROUP_COLORS = {
    "Alkali": "#4fc3f7",
    "Alkaline": "#00e5ff",
    "Lanthanoid": "#ce93d8",
    "TM": "#ffb74d",
    "Post-TM": "#81c784",
    "Metalloid": "#f48fb1",
    "Halogen": "#ef5350",
    "Noble gas": "#c6ff00",
    "Chalcogen": "#ffab91",
    "Others": "#90a4ae",
    "Actinoid": "#b39ddb",
}


def create_3d_map(
    embedding_name: str,
    display_name: str,
    reducer: str = "umap",
) -> None:
    """Create a single striking 3D embedding map."""
    emb = Embedding.load_data(embedding_name)

    # Get reduced coordinates
    if reducer == "umap":
        reduced = emb.calculate_umap(n_components=3)
    elif reducer == "tsne":
        reduced = emb.calculate_tsne(n_components=3)
    else:
        reduced = emb.calculate_pca(n_components=3)

    elements = emb.element_list
    groups = emb.element_groups_dict

    # Set up figure with dark theme
    fig = plt.figure(figsize=(10, 10), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG_COLOR)

    # Plot each element group separately for legend
    plotted_groups = {}
    for i, el in enumerate(elements):
        group = groups[el]
        color = GROUP_COLORS.get(group, "#888888")
        rgba = to_rgba(color, alpha=0.9)

        if group not in plotted_groups:
            ax.scatter(
                reduced[i, 0],
                reduced[i, 1],
                reduced[i, 2],
                c=[rgba],
                s=120,
                edgecolors="white",
                linewidths=0.3,
                depthshade=True,
                zorder=5,
                label=group,
            )
            plotted_groups[group] = True
        else:
            ax.scatter(
                reduced[i, 0],
                reduced[i, 1],
                reduced[i, 2],
                c=[rgba],
                s=120,
                edgecolors="white",
                linewidths=0.3,
                depthshade=True,
                zorder=5,
            )

        # Add element labels
        ax.text(
            reduced[i, 0],
            reduced[i, 1],
            reduced[i, 2],
            f"  {el}",
            fontsize=6,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=10,
        )

    # Style the axes
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("", fontsize=0)
    ax.set_zlabel("", fontsize=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(colors=GRID_COLOR, length=0)

    # Transparent panes and subtle grid
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.3

    # Set a good viewing angle
    ax.view_init(elev=20, azim=135)

    # Title
    reducer_label = reducer.upper()
    ax.set_title(
        f"{display_name} Element Embedding ({reducer_label})",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        pad=0,
        y=0.95,
    )

    # Legend
    legend = ax.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        facecolor=BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        ncol=2,
        borderpad=0.8,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    legend.get_frame().set_alpha(0.8)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fname = f"{embedding_name}_{reducer}_3d.png"
    fig.savefig(
        OUTPUT_DIR / fname,
        dpi=300,
        facecolor=BG_COLOR,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    for reducer in REDUCERS:
        print(f"\n--- {reducer.upper()} ---")
        for emb_name, display_name in EMBEDDINGS.items():
            try:
                create_3d_map(emb_name, display_name, reducer=reducer)
            except Exception as e:
                print(f"  SKIP {emb_name}/{reducer}: {e}")
