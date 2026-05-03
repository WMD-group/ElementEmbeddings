"""Compare all embedding schemes and generate visualisations.

Produces:
- Pairwise correlation heatmap between all embedding schemes
- 2D UMAP map of embedding schemes (each point = one embedding)
- Mantel test results table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from elementembeddings.compare import (
    _get_similarity_matrix,
    _upper_triangle,
    mantel_test,
    pairwise_embedding_comparison,
)
from elementembeddings.core import Embedding

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDINGS = {
    "Magpie": "magpie_sc",
    "Mat2Vec": "mat2vec",
    "MEGNet": "megnet16",
    "SkipAtom": "skipatom",
    "Oliynyk": "oliynyk_sc",
    "XenonPy": "xenonpy",
    "CGNF": "cgnf",
    "CrystaLLM": "crystallm",
    "MACE-MP-0": "mace_mp0",
    "SevenNet": "sevennet",
    "ORB-v2": "orb_v2",
}


def load_all() -> dict[str, Embedding]:
    """Load all embedding schemes."""
    embeddings = {}
    for display_name, code_name in EMBEDDINGS.items():
        print(f"  Loading {display_name}...")
        embeddings[display_name] = Embedding.load_data(code_name)
    return embeddings


def plot_comparison_heatmap(comparison_df: pd.DataFrame) -> None:
    """Plot the pairwise embedding comparison as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        comparison_df,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Embedding Scheme Similarity\n(Pearson correlation of cosine similarity matrices)", fontsize=12)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "embedding_comparison_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved embedding_comparison_heatmap.png")


def plot_embedding_map(embeddings: dict[str, Embedding]) -> None:
    """Plot embeddings in 2D using UMAP on flattened similarity vectors."""
    # Find common elements across ALL embeddings
    all_elements = None
    for emb in embeddings.values():
        els = set(emb.element_list)
        all_elements = els if all_elements is None else all_elements & els
    common = sorted(all_elements)
    print(f"  Common elements for UMAP: {len(common)}")

    # Build feature matrix: each row = one embedding's flattened similarity
    names = list(embeddings.keys())
    vectors = []
    for name in names:
        mat = _get_similarity_matrix(embeddings[name], common, "cosine_similarity")
        vectors.append(_upper_triangle(mat))

    X = np.array(vectors)
    reduced = UMAP(n_components=2, random_state=42, n_neighbors=5).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], s=120, c="steelblue", edgecolors="white", linewidths=0.5, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (reduced[i, 0], reduced[i, 1]),
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embedding Schemes in 2D\n(UMAP of flattened cosine similarity matrices)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "embedding_umap_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved embedding_umap_map.png")


def run_mantel_tests(embeddings: dict[str, Embedding]) -> None:
    """Run Mantel tests between selected embedding pairs."""
    pairs = [
        ("Magpie", "Oliynyk"),
        ("Mat2Vec", "SkipAtom"),
        ("MACE-MP-0", "SevenNet"),
        ("MACE-MP-0", "ORB-v2"),
        ("SevenNet", "ORB-v2"),
        ("Mat2Vec", "MACE-MP-0"),
        ("Magpie", "MACE-MP-0"),
    ]

    results = []
    for name1, name2 in pairs:
        print(f"  Mantel: {name1} vs {name2}...")
        r, p = mantel_test(embeddings[name1], embeddings[name2], n_permutations=999)
        results.append({"Embedding 1": name1, "Embedding 2": name2, "r": round(r, 4), "p-value": round(p, 4)})

    df = pd.DataFrame(results)
    print("\nMantel test results:")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "mantel_test_results.csv", index=False)
    print("Saved mantel_test_results.csv")


if __name__ == "__main__":
    embeddings = load_all()

    print("\nComputing pairwise comparisons...")
    comparison_df = pairwise_embedding_comparison(embeddings)
    comparison_df.to_csv(OUTPUT_DIR / "pairwise_comparison.csv")
    print("Saved pairwise_comparison.csv")

    print("\nPlotting comparison heatmap...")
    plot_comparison_heatmap(comparison_df)

    print("\nPlotting embedding UMAP map...")
    plot_embedding_map(embeddings)

    print("\nRunning Mantel tests...")
    run_mantel_tests(embeddings)
