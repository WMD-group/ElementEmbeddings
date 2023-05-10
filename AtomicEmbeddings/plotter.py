"""Provides the plotting functions for visualising Embeddings."""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        p = embedding.correlation_pivot_table(metric=metric, sortby=sortaxisby)

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


def dimension_plotter(
    embedding: Embedding,
    ax: Optional[plt.axes] = None,
    n_components: int = 2,
    reducer: str = "umap",
    **kwargs,
):
    """Plot the reduced dimensions of the embeddings.

    Parameters
    ----------
    embedding : Embedding
        The embedding to be plotted.
    ax : Optional[plt.axes], optional
        The axes to plot on, by default None
    n_components : int, optional
        The number of components to reduce to, by default 2
    reducer : str, optional
        The dimensionality reduction algorithm to use, by default "umap"
    **kwargs
        Additional keyword arguments to pass to the dimensionality reduction algorithm.

    """
    if reducer == "umap":
        if embedding._umap_data and embedding._umap_data.shape[1] == n_components:
            reduced = embedding._umap_data
        else:
            reduced = embedding.calculate_UMAP(n_components=n_components, **kwargs)
    elif reducer == "tsne":
        if embedding._tsne_data and embedding._tsne_data.shape[1] == n_components:
            reduced = embedding._tsne_data
        else:
            reduced = embedding.calculate_tSNE(n_components=n_components, **kwargs)
    elif reducer == "pca":
        if embedding._pca_data and embedding._tsne_data.shape[1] == n_components:
            reduced = embedding._pca_data
        else:
            reduced = embedding.calculate_tSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError("Unrecognised reducer.")

    if reduced.shape[1] == 2:
        df = pd.DataFrame(
            {
                "x": reduced[:, 0],
                "y": reduced[:, 1],
                "element": np.array(embedding.element_list),
                "group": list(embedding.element_groups_dict.values()),
            }
        )
        if not ax:
            fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="x", y="y", hue="group", ax=ax, **kwargs)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        for i in range(len(df)):
            plt.text(df["x"][i], df["y"][i], df["element"][i], fontsize=12)
    elif reduced.shape[1] == 3:
        df = pd.DataFrame(
            {
                "x": reduced[:, 0],
                "y": reduced[:, 1],
                "z": reduced[:, 2],
                "element": np.array(embedding.element_list),
                "group": list(embedding.element_groups_dict.values()),
            }
        )
        if not ax:
            fig = plt.figure()  # noqa: F841
            ax = plt.axes(projection="3d")
        ax.scatter3D(
            df["x"],
            df["y"],
            df["z"],
            # c=df["group"],
            # **kwargs
        )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        for i in range(len(df)):
            ax.text(df["x"][i], df["y"][i], df["z"][i], df["element"][i], fontsize=12)
    else:
        raise ValueError("Unrecognised number of dimensions.")
    ax.set_title(embedding.embedding_name, fontdict={"fontweight": "bold"})
    # fig.tight_layout()
    return ax


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
