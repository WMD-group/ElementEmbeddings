"""Generate composite image: high-dimensional embedding heatmap + 3D map.

Shows the raw embedding vectors as a coloured heatmap strip (the "string"
of numbers for each element) alongside the 3D reduced map, conveying
the dimensionality reduction from hundreds of dimensions to three.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec

from elementembeddings.core import Embedding
from elementembeddings.utils.config import ELEMENT_GROUPS_PALETTES

OUTPUT_DIR = Path(__file__).parent / "3d_maps"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dark theme
BG_COLOR = "#0a0a1a"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#1a1a3a"
ACCENT = "#e0e0e0"

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

# Embeddings to concatenate — chosen for variety and visual impact
CONCAT_EMBEDDINGS = [
    ("mat2vec", "Mat2Vec (200D)"),
    ("skipatom", "SkipAtom (200D)"),
    ("matscholar", "Matscholar (200D)"),
    ("megnet16", "MEGNet (16D)"),
    ("magpie", "Magpie (22D)"),
]

# Which single embedding to use for the 3D map
MAP_EMBEDDING = "mat2vec"
MAP_NAME = "Mat2Vec"
REDUCER = "umap"


def build_concat_matrix():
    """Load and concatenate multiple embeddings into one wide matrix."""
    embeddings = []
    labels = []
    section_widths = []
    section_labels = []

    # Find common elements across all embeddings
    loaded = []
    for emb_name, display_name in CONCAT_EMBEDDINGS:
        emb = Embedding.load_data(emb_name)
        emb.standardise(inplace=True)
        loaded.append((emb, display_name))

    common = set(loaded[0][0].element_list)
    for emb, _ in loaded[1:]:
        common &= set(emb.element_list)

    # Use first embedding's order, filtered to common elements
    ref = loaded[0][0]
    elements = [el for el in ref.element_list if el in common]
    groups = ref.element_groups_dict

    for emb, display_name in loaded:
        mat = np.array([emb.embeddings[el] for el in elements])
        embeddings.append(mat)
        section_widths.append(mat.shape[1])
        section_labels.append(display_name)

    concat = np.hstack(embeddings)
    return concat, elements, groups, section_widths, section_labels


def create_composite():
    """Create the composite figure: heatmap left, 3D map right."""
    concat, elements, groups, section_widths, section_labels = build_concat_matrix()
    total_dims = concat.shape[0]

    # Load embedding for 3D map
    emb = Embedding.load_data(MAP_EMBEDDING)
    reduced = emb.calculate_umap(n_components=3)

    # --- Figure layout ---
    fig = plt.figure(figsize=(20, 16), facecolor=BG_COLOR)
    gs = GridSpec(
        1, 2,
        width_ratios=[1, 1.1],
        wspace=0.08,
        left=0.04, right=0.96, top=0.90, bottom=0.06,
    )

    # ====== LEFT: Heatmap ======
    ax_heat = fig.add_subplot(gs[0], facecolor=BG_COLOR)

    # Clip extremes for visual clarity
    vmax = 3.0
    im = ax_heat.imshow(
        concat,
        aspect="auto",
        cmap="inferno",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # Element labels on left, colored by group
    ax_heat.set_yticks(range(len(elements)))
    ax_heat.set_yticklabels(elements, fontsize=7, fontfamily="monospace")
    for i, el in enumerate(elements):
        color = GROUP_COLORS.get(groups[el], "#888888")
        ax_heat.get_yticklabels()[i].set_color(color)

    # Section dividers and labels at top
    cumulative = 0
    for width, label in zip(section_widths, section_labels):
        mid = cumulative + width / 2
        ax_heat.text(
            mid, -2.5, label,
            ha="center", va="bottom",
            fontsize=9, color=ACCENT,
            fontweight="bold",
            rotation=0,
        )
        if cumulative > 0:
            ax_heat.axvline(cumulative - 0.5, color=BG_COLOR, linewidth=1.5)
        cumulative += width

    ax_heat.set_xlim(-0.5, concat.shape[1] - 0.5)
    ax_heat.set_xticks([])
    ax_heat.tick_params(axis="y", length=0, pad=2)
    ax_heat.spines[:].set_visible(False)

    # Dimension count annotation
    ax_heat.set_xlabel(
        f"{concat.shape[1]} dimensions",
        color=ACCENT, fontsize=14, fontweight="bold", labelpad=10,
    )

    # ====== RIGHT: 3D Map ======
    ax_3d = fig.add_subplot(gs[1], projection="3d", facecolor=BG_COLOR)

    plotted_groups = {}
    for i, el in enumerate(elements):
        group = groups[el]
        color = GROUP_COLORS.get(group, "#888888")
        rgba = to_rgba(color, alpha=0.9)

        kwargs = {
            "c": [rgba],
            "s": 100,
            "edgecolors": "white",
            "linewidths": 0.3,
            "depthshade": True,
            "zorder": 5,
        }
        if group not in plotted_groups:
            kwargs["label"] = group
            plotted_groups[group] = True

        ax_3d.scatter(reduced[i, 0], reduced[i, 1], reduced[i, 2], **kwargs)

        ax_3d.text(
            reduced[i, 0], reduced[i, 1], reduced[i, 2],
            f"  {el}",
            fontsize=7, color=color, fontweight="bold",
            ha="left", va="center", zorder=10,
        )

    # Style 3D axes
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.set_xlabel("")
    ax_3d.set_ylabel("")
    ax_3d.set_zlabel("")
    ax_3d.tick_params(colors=GRID_COLOR, length=0)

    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.3

    ax_3d.view_init(elev=20, azim=135)

    ax_3d.set_xlabel("3 dimensions", color=ACCENT, fontsize=14, fontweight="bold", labelpad=14)

    # Legend
    legend = ax_3d.legend(
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

    # ====== Titles ======
    fig.text(
        0.27, 0.95,
        "Element Embedding Vectors",
        ha="center", va="bottom",
        fontsize=18, fontweight="bold", color=TEXT_COLOR,
    )
    fig.text(
        0.73, 0.95,
        f"3D Map ({MAP_NAME}, {REDUCER.upper()})",
        ha="center", va="bottom",
        fontsize=18, fontweight="bold", color=TEXT_COLOR,
    )

    # Arrow between panels
    fig.text(
        0.50, 0.50, "\u2192",
        ha="center", va="center",
        fontsize=42, color=ACCENT, fontweight="bold",
    )

    fname = "composite_embedding_map.png"
    fig.savefig(
        OUTPUT_DIR / fname,
        dpi=300,
        facecolor=BG_COLOR,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    create_composite()
