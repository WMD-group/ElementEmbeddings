"""Provides the plotting functions for visualising Embeddings."""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from .core import Embedding


def heatmap_plotter(
    embedding: Embedding,
    metric: str,
    cmap: str = "Blues",
    sortaxisby: str = "mendeleev",
    ax: Optional[plt.axes] = None,
    show_axislabels: bool = True,
    **kwargs,
):
    """
    Plot multiple heatmaps of the embeddings.

    Parameters
    ----------
    embedding : Embedding
        The embeddings to be plotted.
    metric : str
        The distance metric / similarity measure to be plotted.
    cmap: str
        The colourmap for the heatmap.
    sortaxisby : str, optional
        The attribute to sort the axis by, by default "mendeleev_number".
        Options are "mendeleev_number", "atomic_number"
    ax : Optional[plt.axes], optional
        The axes to plot on, by default None
    show_axislabels : bool, optional
        Whether to show the axis, by default True
    **kwargs
        Additional keyword arguments to pass to seaborn.heatmap

    """
    if not ax:
        fig, ax = plt.subplots()

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
        p = embedding.correlation_pivot_table()

    elif metric in distance_metrics:
        p = embedding.distance_pivot_table(metric=metric, sortby=sortaxisby)
    xlabels = [i[1] for i in p.index]
    ylabels = [i[1] for i in p.columns]
    sns.heatmap(
        p,
        cmap=cmap,
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
