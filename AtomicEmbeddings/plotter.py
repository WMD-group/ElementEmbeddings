"""Provides the plotting functions for visualising Embeddings."""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from .core import Embedding


def heatmap_plotter(
    embedding: Embedding,
    metric: str,
    distance: bool = True,
    correlation: bool = False,
    figsize: Tuple[int, int] = (36, 24),
    filename: Optional[str] = None,
    show: bool = True,
    **kwargs,
):
    """
    Plot a heatmap of the embedding.

    Parameters
    ----------
    embedding : Embedding
        The embedding to be plotted.
    metric : str
        The distance/correlation metric to be used.
    distance : bool, optional
        Whether to plot a distance heatmap, by default True
    correlation : bool, optional
        Whether to plot a correlation heatmap, by default False
    figsize : Tuple[int,int], optional
        The size of the figure, by default (36, 24)
    filename : Optional[str], optional
        The filename to save the figure to, by default None
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to seaborn.heatmap

    """
    fig, ax = plt.subplots(figsize=figsize)
    if correlation:
        pivot = embedding.pearson_pivot_table()

    elif distance:
        pivot = embedding.distance_pivot_table(metric=metric)
    xlabels = [i[1] for i in pivot.index]
    ylabels = [i[1] for i in pivot.columns]
    sns.heatmap(
        pivot,
        cmap="bwr",
        square="True",
        linecolor="k",
        ax=ax,
        cbar_kws={"shrink": 0.5},
        xticklabels=True,
        yticklabels=True,
        **kwargs,
    )
    ax.title.set_text(embedding.embedding_name)
    ax.set_xticklabels(
        xlabels,
    )
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()


def multi_heatmap_plotter(
    embeddings: List[Embedding],
    nrows: int,
    ncols: int,
    metric: str,
    sortaxisby: str = "mendeleev",
    figsize: Tuple[int, int] = (36, 36),
    filename: Optional[str] = None,
    show_axislabels: bool = True,
    show_plot: bool = True,
    **kwargs,
):
    """
    Plot multiple heatmaps of the embeddings.

    Parameters
    ----------
    embeddings : List[Embedding]
        The embeddings to be plotted.
    nrows : int
        The number of rows in the figure.
    ncols : int
        The number of columns in the figure.
    metric : bool, optional
        Whether to plot a metric distance heatmap, by default False
    sortaxisby : str, optional
        The attribute to sort the axis by, by default "mendeleev_number".
        Options are "mendeleev_number", "atomic_number"
    figsize : Tuple[int,int], optional
        The size of the figure, by default (36, 36)
    filename : Optional[str], optional
        The filename to save the figure to, by default None
    show_axislabels : bool, optional
        Whether to show the axis, by default True
    show_plot : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to seaborn.heatmap

    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, embedding in enumerate(embeddings):
        ax = axes[i // ncols, i % ncols]

        correlation_metrics = ["spearman", "pearson", "cosine_similarity"]
        distance_metrics = [
            "euclidean",
            "manhattan",
            "cosine_distance",
            "chebyshev",
            "wasserstein",
            "energy",
        ]
        if metric in correlation_metrics:
            p = embedding.pearson_pivot_table()

        elif metric in distance_metrics:
            p = embedding.distance_pivot_table(metric=metric, sortby=sortaxisby)
        xlabels = [i[1] for i in p.index]
        ylabels = [i[1] for i in p.columns]
        sns.heatmap(
            p,
            cmap="bwr",
            square="True",
            linecolor="k",
            ax=ax,
            cbar_kws={
                "shrink": 0.5,
            },
            xticklabels=True,
            yticklabels=True,
            **kwargs,
        )
        ax.set_title(
            embedding.embedding_name,
            fontdict={
                # "fontsize": 30,
                "fontweight": "bold"
            },
        )
        if not show_axislabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticklabels(
                xlabels,
            )
            ax.set_yticklabels(ylabels)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    if show_plot:
        plt.show()
