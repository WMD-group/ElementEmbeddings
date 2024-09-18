"""Provides the plotting functions for visualising Embeddings."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from .core import Embedding, SpeciesEmbedding
from .utils.config import ELEMENT_GROUPS_PALETTES
from .utils.species import get_sign, parse_species


def heatmap_plotter(
    embedding: Embedding | SpeciesEmbedding,
    metric: str,
    cmap: str = "Blues",
    sortaxisby: str = "mendeleev",
    ax: plt.axes | None = None,
    show_axislabels: bool = True,
    **kwargs,
):
    """Plot multiple heatmaps of the embeddings.

    Args:
    ----
        embedding (Embedding): The embeddings to be plotted.
        metric (str): The distance metric / similarity measure to be plotted.
        cmap (str): The colourmap for the heatmap.
        sortaxisby (str, optional): The attribute to sort the axis by,
        by default "mendeleev_number".
        Options are "mendeleev_number", "atomic_number"
        ax (plt.axes, optional): The axes to plot on, by default None
        show_axislabels (bool, optional): Whether to show the axis, by default True
        **kwargs: Additional keyword arguments to pass to seaborn.heatmap

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
    else:
        raise ValueError("Unrecognised metric.")
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
            "fontweight": "bold",
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
    return ax


def dimension_plotter(
    embedding: Embedding | SpeciesEmbedding,
    ax: plt.axes | None = None,
    n_components: int = 2,
    reducer: str = "umap",
    adjusttext: bool = True,
    reducer_params: dict | None = None,
    scatter_params: dict | None = None,
    include_species: list | None = None,
):
    """Plot the reduced dimensions of the embeddings.

    Args:
    ----
        embedding (Embedding): The embedding to be plotted.
        ax (plt.axes, optional): The axes to plot on, by default None
        n_components (int): The number of components to reduce to, by default 2
        reducer (str): The dimensionality reduction algorithm to use, by default "umap"
        adjusttext (bool): Whether to avoid overlap of the text labels, by default True
        reducer_params (dict, optional): Additional keyword arguments to pass to
        the reducer, by default None
        scatter_params (dict, optional): Additional keyword arguments to pass to
        the scatterplot, by default None
        include_species (list, optional): The elements/species to include in the plot,

    """
    if reducer_params is None:
        reducer_params = {}
    if reducer == "umap":
        reduced = embedding.calculate_umap(n_components=n_components, **reducer_params)
    elif reducer == "tsne":
        reduced = embedding.calculate_tsne(n_components=n_components, **reducer_params)
    elif reducer == "pca":
        reduced = embedding.calculate_pca(n_components=n_components, **reducer_params)
    else:
        msg = "Unrecognised reducer."
        raise ValueError(msg)

    if isinstance(embedding, Embedding):
        group_dict = embedding.element_groups_dict
        el_sp_array = np.array(embedding.element_list)

        data = {
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "element": el_sp_array,
            "Group": list(group_dict.values()),
        }
    elif isinstance(embedding, SpeciesEmbedding):
        group_dict = embedding.species_groups_dict
        el_sp_array = np.array(embedding.species_list)
        ion_type = embedding.ion_type_dict
        data = {
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "element": el_sp_array,
            "Group": list(group_dict.values()),
            "ion_type": list(ion_type.values()),
        }
    if reduced.shape[1] == 2:
        df = pd.DataFrame(data)
        if include_species:
            df = df[df["element"].isin(include_species)].reset_index(drop=True)
        if not ax:
            fig, ax = plt.subplots()
        if scatter_params is None:
            scatter_params = {}
        if isinstance(embedding, SpeciesEmbedding):
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="Group",
                ax=ax,
                palette=ELEMENT_GROUPS_PALETTES,
                style="ion_type",
                **scatter_params,
            )
            # Convert the species to (element, charge) format
            parsed_species = [parse_species(spec) for spec in df["element"].tolist()]
            signs = [get_sign(charge) for _, charge in parsed_species]

            species_labels = [
                rf"$\mathregular{{{element}^{{{abs(charge)}{sign}}}}}$"
                for (element, charge), sign in zip(parsed_species, signs)
            ]

            texts = [ax.text(df["x"][i], df["y"][i], species_labels[i], fontsize=12) for i in range(len(df))]
        elif isinstance(embedding, Embedding):
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="Group",
                ax=ax,
                palette=ELEMENT_GROUPS_PALETTES,
                **scatter_params,
            )
            texts = [ax.text(df["x"][i], df["y"][i], df["element"][i], fontsize=12) for i in range(len(df))]
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        if adjusttext:
            adjust_text(
                texts,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
                ax=ax,
            )

    elif reduced.shape[1] == 3:
        df = pd.DataFrame(
            {
                "x": reduced[:, 0],
                "y": reduced[:, 1],
                "z": reduced[:, 2],
                "element": el_sp_array,
                "group": list(group_dict.values()),
            },
        )
        if include_species:
            df = df[df["element"].isin(include_species)].reset_index(drop=True)
        if not ax:
            fig = plt.figure()  # noqa: F841
            ax = plt.axes(projection="3d")
        ax.scatter3D(
            df["x"],
            df["y"],
            df["z"],
        )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        for i in range(len(df)):
            ax.text(df["x"][i], df["y"][i], df["z"][i], df["element"][i], fontsize=12)
    else:
        msg = "Unrecognised number of dimensions."
        raise ValueError(msg)
    ax.set_title(embedding.embedding_name, fontdict={"fontweight": "bold"})
    return ax
