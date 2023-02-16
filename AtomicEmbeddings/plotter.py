"""Provides the plotting functions for visualising Embeddings."""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from .core import Embedding


def heatmap_plotter(
    embedding: Embedding,
    metric: bool = False,
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
    metric : bool, optional
        Whether to plot a metric distance heatmap, by default False
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
        p = embedding.pearson_pivot_table()

    elif distance:
        p = embedding.distance_pivot_table(metric=metric)
    xlabels = [i[1] for i in p.index]
    ylabels = [i[1] for i in p.columns]
    sns.heatmap(
        p,
        cmap="bwr",
        square="True",
        linecolor="k",
        ax=ax,
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
        plt.savefig("plots/" + filename)
    if show:
        plt.show()
