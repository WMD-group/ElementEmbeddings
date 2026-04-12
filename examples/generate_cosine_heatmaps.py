"""Generate cosine similarity heatmaps for all embeddings + animated GIF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from PIL import Image

from elementembeddings.core import Embedding
from elementembeddings.plotter import heatmap_plotter

OUTPUT_DIR = Path(__file__).parent / "2d_cosine"
OUTPUT_DIR.mkdir(exist_ok=True)

BG_COLOR = "white"
TEXT_COLOR = "#222222"

SUPERHEAVY = [
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

ALL_EMBEDDINGS = [
    ("magpie", "Magpie"),
    ("mat2vec", "Mat2Vec"),
    ("megnet16", "MEGNet"),
    ("skipatom", "SkipAtom"),
    ("oliynyk", "Oliynyk"),
    ("xenonpy", "XenonPy"),
    ("cgnf", "CGNF"),
    ("crystallm", "CrystaLLM"),
    ("magpie_sc", "Magpie (scaled)"),
    ("oliynyk_sc", "Oliynyk (scaled)"),
    ("mace_mp0", "MACE-MP-0"),
    ("sevennet", "SevenNet"),
    ("orb_v2", "ORB-v2"),
]

GIF_EMBEDDINGS = [
    ("magpie_sc", "Magpie"),
    ("mat2vec", "Mat2Vec"),
    ("megnet16", "MEGNet"),
    ("skipatom", "SkipAtom"),
    ("oliynyk_sc", "Oliynyk"),
    ("xenonpy", "XenonPy"),
    ("cgnf", "CGNF"),
    ("crystallm", "CrystaLLM"),
    ("mace_mp0", "MACE-MP-0"),
    ("sevennet", "SevenNet"),
    ("orb_v2", "ORB-v2"),
]

gif_set = {name for name, _ in GIF_EMBEDDINGS}
gif_labels = dict(GIF_EMBEDDINGS)

# Find common elements across all embeddings for consistent axes
print("Finding common elements...")
common_elements = None
for emb_name, _ in ALL_EMBEDDINGS:
    emb = Embedding.load_data(emb_name)
    els = set(emb.element_list) - set(SUPERHEAVY)
    common_elements = els if common_elements is None else common_elements & els
print(f"  {len(common_elements)} common elements across all embeddings")

frames = []

for emb_name, display_name in ALL_EMBEDDINGS:
    print(f"  {display_name}...")
    emb = Embedding.load_data(emb_name)

    # Remove elements not in common set for consistent axes
    to_remove = [el for el in emb.element_list if el not in common_elements]
    if to_remove:
        emb.remove_elements(to_remove, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 14), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    heatmap_plotter(
        emb,
        metric="cosine_similarity",
        cmap="inferno",
        sortaxisby="mendeleev",
        ax=ax,
        show_axislabels=True,
        linewidths=0,
        vmin=0,
        vmax=1,
    )

    ax.tick_params(colors=TEXT_COLOR, labelsize=5.5, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cbar.set_label("Cosine Similarity", color=TEXT_COLOR, fontsize=11)

    ax.set_title("")

    # Save static PNG (high-res, no label)
    fig.savefig(
        OUTPUT_DIR / f"{emb_name}_cosine.png",
        dpi=300,
        facecolor=BG_COLOR,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # Add GIF frame for selected embeddings
    if emb_name in gif_set:
        ax.text(
            0.97,
            0.97,
            gif_labels[emb_name],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=20,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=3, foreground="#222222")],
        )

        frame_path = OUTPUT_DIR / f"_frame_{emb_name}.png"
        fig.savefig(
            frame_path,
            dpi=150,
            facecolor=BG_COLOR,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        frames.append(Image.open(frame_path))

    plt.close(fig)

# Build GIF
gif_path = OUTPUT_DIR / "cosine_similarity_all_embeddings.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=1200,
    loop=0,
)

# Clean up frame PNGs
for f in OUTPUT_DIR.glob("_frame_*.png"):
    f.unlink()

print(f"\nSaved {len(ALL_EMBEDDINGS)} static PNGs and GIF ({len(frames)} frames) to {OUTPUT_DIR}")
