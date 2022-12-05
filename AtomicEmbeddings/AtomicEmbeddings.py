import json
import os
import random
from itertools import combinations_with_replacement
from os import path
from typing import Callable, Generator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
from pymatgen.core import Element
from scipy.stats import energy_distance, pearsonr, wasserstein_distance
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import DistanceMetric

""" Provides the `Atomic_Embeddings` class.

This module enables the user load in elemetal representation data and analyse it using statistical functions.

Typical usage example:

    megnet16 = Atomic_Embeddings.from_json('megnet16')
"""

module_directory = path.abspath(path.dirname(__file__))
data_directory = path.join(module_directory, "data")


class Atomic_Embeddings:

    """
    Represents an elemental representation, which is essentially a dictionary of {element: vector} pairs.

    Works like a standard python dictionary.

    Adds a few convenience methods related to elemental representations.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

        # Grab a random value from the embedding vector
        _rand_embed = random.choice(list(self.embeddings.values()))
        # Convert embeddings to numpy array if not already a numpy array
        if not isinstance(_rand_embed, np.ndarray):
            self.embeddings = {
                ele: np.array(self.embeddings[ele]) for ele in self.embeddings
            }
        # Determines if the embedding vector has a length attribute
        # (i.e. is not a scalar int or float)
        # If the 'vector' is a scalar/float, the representation is linear (dim=1)
        if hasattr(_rand_embed, "__len__") and (not isinstance(_rand_embed, str)):
            self.dim = len(random.choice(list(self.embeddings.values())))
        else:
            self.dim = 1

    @staticmethod
    def from_json(embedding_json: Optional[str] = None):
        """Creates an instance of the `Atomic_Embeddings` class from a default embedding file.

        The default embeddings are in the table below:

        | **Name**                | **str_name** |
        |-------------------------|--------------|
        | Magpie                  | magpie       |
        | Magpie (scaled)         | magpie_sc    |
        | Mat2Vec                 | mat2vec      |
        | Matscholar              | matscholar   |
        | Megnet (16 dimensions)  | megnet16     |
        | Modified pettifor scale | mod_petti    |
        | Oliynyk                 | oliynyk      |
        | Oliynyk (scaled)        | oliynyk_sc   |
        | Random (200 dimensions) | random_200   |
        | SkipAtom                | skipatom     |


        Args:
            embedding_json (str): JSON-style representation of a set of atomic embedding vectors. This is a python dictionary of element:embedding vector pairs.

        Returns:

            Atomic_Embedding :class:`Atomic_Embeddings` instance."""

        _cbfv_files = {
            "magpie": "magpie.json",
            "magpie_sc": "magpie_sc.json",
            "mat2vec": "mat2vec.json",
            "matscholar": "matscholar-embedding.json",
            "megnet16": "megnet16.json",
            "mod_petti": "mod_petti.json",
            "oliynyk": "oliynyk.json",
            "oliynyk_sc": "oliynyk_sc.json",
            "random_200": "random_200.csv",
            "skipatom": "skipatom_20201009_induced.csv",
        }
        _cbfv_names = list(_cbfv_files.keys())
        _cbfv_names_others = [
            i for i in _cbfv_names if i not in ["skipatom", "random_200", "megnet16"]
        ]

        # Get the embeddings
        if embedding_json in _cbfv_files:
            if embedding_json == "skipatom" or embedding_json == "random_200":
                _csv = path.join(data_directory, _cbfv_files[embedding_json])
                df = pd.read_csv(_csv)
                # Convert df to a dictionary of (ele:embeddings) pairs
                elements = list(df["element"])
                df.drop(["element"], axis=1, inplace=True)
                embeds_array = df.to_numpy()
                embedding_data = {
                    elements[i]: embeds_array[i] for i in range(len(embeds_array))
                }

            elif embedding_json == "megnet16":
                megnet16_json = path.join(data_directory, _cbfv_files["megnet16"])
                with open(megnet16_json, "r") as f:
                    embedding_data = json.load(f)
                # Remove 'Null' key from megnet embedding
                del embedding_data["Null"]

            elif embedding_json in _cbfv_names_others:
                _json = path.join(data_directory, _cbfv_files[embedding_json])
                with open(_json, "r") as f:
                    embedding_data = json.load(f)

            # Load a json file from a file specified in the input
            else:
                with open(embedding_json, "r") as f:
                    embedding_data = json.load(f)
        else:
            raise (
                ValueError(
                    f"{embedding_json} not in the data directory or not in directory."
                )
            )
        return Atomic_Embeddings(embedding_data)

    @property
    def element_list(self):
        """Returns the elements of the atomic embedding."""
        return list(self.embeddings.keys())

    @property
    def element_groups_dict(self):
        """Returns a dictionary of {element: element type} pairs e.g. {'He':'Noble gas'}"""

        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[i] for i in self.element_list}

    def create_pairs(self):
        """Creates all possible pairs of elements"""
        ele_list = self.element_list
        ele_pairs = combinations_with_replacement(ele_list, 2)
        return ele_pairs

    def create_correlation_df(self):
        """Returns a pandas.DataFrame object with columns of the elements and correlation metrics"""
        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            pearson = pearsonr(self.embeddings[ele1], self.embeddings[ele2])
            dist = norm(self.embeddings[ele1] - self.embeddings[ele2])

            recip_dist = dist**-1
            table.append((ele1, ele2, pearson[0], dist, recip_dist))
            if ele1 != ele2:
                table.append((ele2, ele1, pearson[0], dist, recip_dist))

        corr_df = pd.DataFrame(
            table,
            columns=[
                "ele_1",
                "ele_2",
                "pearson_corr",
                "euclid_dist",
                "reciprocal_euclid_dist",
            ],
        )

        mend_1 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_1"]]
        mend_2 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df = corr_df[
            [
                "ele_1",
                "ele_2",
                "mend_1",
                "mend_2",
                "euclid_dist",
                "reciprocal_euclid_dist",
                "pearson_corr",
            ]
        ]

        return corr_df

    def compute_distance_metric(self, ele1, ele2, metric="euclidean"):
        """Computes distance metric between two vectors.

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy


        Args:
            ele1 (str): element symbol
            ele2 (str): element symbol
            metric (str): name of a distance metric

        Returns:
            distance (float): distance between embedding vectors
        """

        # Define the allowable metrics
        scikit_metrics = ["euclidean", "manhattan", "chebyshev"]

        scipy_metrics = {"wasserstein": wasserstein_distance, "energy": energy_distance}

        valid_metrics = scikit_metrics + list(scipy_metrics.keys())

        # Validate if the elements are within the embedding vector
        if ele1 not in self.element_list:
            print("ele1 is not an element included within the atomic embeddings")
            raise ValueError
        if ele2 not in self.element_list:
            print("ele2 is not an element included within the atomic embeddings")
            raise ValueError

        # Compute the distance measure
        if metric in scikit_metrics:
            distance = DistanceMetric.get_metric(metric)

            return distance.pairwise(
                self.embeddings[ele1].reshape(1, -1),
                self.embeddings[ele2].reshape(1, -1),
            )[0][0]

        elif metric in scipy_metrics.keys():
            return scipy_metrics[metric](self.embeddings[ele1], self.embeddings[ele2])

        else:
            print(
                f"Invalid distance metric. Use one of the following metrics:{valid_metrics}"
            )
            raise ValueError

    def create_pearson_pivot_table(self):
        """Returns a pandas.DataFrame style pivot with the index and column being the mendeleev number of the element pairs and the values being the pearson correlation metrics"""

        corr_df = self.create_correlation_df()
        pearson_pivot = corr_df.pivot_table(
            values="pearson_corr", index="mend_1", columns="mend_2"
        )
        return pearson_pivot

    def create_distance_correlation_df(self, metric="euclidean"):
        """Returns a pandas.DataFrame object with columns of the elements and correlation metrics.

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy

        Args:
            metric (str): A distance metric

        Returns:
            df (pandas.DataFrame): A dataframe with columns ["ele_1", "ele_2", metric]
        """

        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            dist = self.compute_distance_metric(ele1, ele2, metric=metric)
            table.append((ele1, ele2, dist))
            if ele1 != ele2:
                table.append((ele2, ele1, dist))
        corr_df = pd.DataFrame(table, columns=["ele_1", "ele_2", metric])

        mend_1 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_1"]]
        mend_2 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df = corr_df[["ele_1", "ele_2", "mend_1", "mend_2", metric]]

        return corr_df

    def create_distance_pivot_table(self, metric="euclidean"):

        """Returns a pandas.DataFrame style pivot with the index and column being the mendeleev number of the element pairs and the values being a user-specified distance metric

        Args:
            metric (str): A distance metric

        Returns:
            distance_pivot (pandas.DataFrame): A pandas DataFrame pivot table where the index and columns are the elements and the values are the pairwise distance metric.
        """

        corr_df = self.create_distance_correlation_df(metric=metric)
        distance_pivot = corr_df.pivot_table(
            values=metric, index="mend_1", columns="mend_2"
        )
        return distance_pivot

    def plot_pearson_correlation(self, figsize=(24, 24), **kwargs):

        """
        Plots the heatmap of the pearson correlation values for the elemental representation.

        Args:
            figsize (tuple): A tuple of (width, height) to pass to the matplotlib.pyplot.figure object
            **kwargs: Other keyword arguments to be passed to sns.heatmap

        Returns:
            ax (matplotlib Axes): An Axes object with the heatmap

        """

        pearson_pivot = self.create_pearson_pivot_table()

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            pearson_pivot, cmap="bwr", square=True, linecolor="k", **kwargs
        )

        return ax

    def plot_distance_correlation(self, metric="euclidean", figsize=(24, 24), **kwargs):

        """
        Plots the heatmap of the pairwise distance metrics for the elemental representation.

        Args:
            metric (str): A valid distance metric
            figsize (tuple): A tuple of (width, height) to pass to the matplotlib.pyplot.figure object

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the heatmap

        """
        distance_pivot = self.create_distance_pivot_table(metric=metric)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            distance_pivot, cmap="bwr", square=True, linecolor="k", **kwargs
        )

        return ax

    def plot_PCA_2D(
        self, figsize=(16, 12), points_hue="group", points_size=200, **kwargs
    ):
        """A function to plot a PCA plot of the atomic embedding.

        Args:
            figsize (tuple): A tuple of (width, height) to pass to the matplotlib.pyplot.figure object
            points_size (float): The marker size

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the PCA plot

        """
        embeddings_array = np.array(list(self.embeddings.values()))
        element_array = np.array(self.element_list)

        fig = plt.figure(figsize=figsize)
        plt.cla()  # clear current axes
        pca = decomposition.PCA(n_components=2)  # project to 2 dimensions

        pca.fit(embeddings_array)
        X = pca.transform(embeddings_array)

        pca_dim1 = X[:, 0]
        pca_dim2 = X[:, 1]

        # Create a dataframe to store the dimensions, labels and group info for the PCA
        pca_df = pd.DataFrame(
            {
                "pca_dim1": pca_dim1,
                "pca_dim2": pca_dim2,
                "element": element_array,
                "group": list(self.element_groups_dict.values()),
            }
        )

        ax = sns.scatterplot(
            x="pca_dim1",
            y="pca_dim2",
            data=pca_df,
            hue=points_hue,
            s=points_size,
            **kwargs,
        )

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        for i in range(len(X)):
            plt.text(x=pca_dim1[i], y=pca_dim2[i], s=element_array[i])

        return plt

    def plot_tSNE(
        self,
        n_components=2,
        figsize=(16, 12),
        points_hue="group",
        points_size=200,
        **kwargs,
    ):
        """A function to plot a t-SNE plot of the atomic embedding

        Args:
            n_components (int): Number of t-SNE components to plot.
            figsize (tuple): A tuple of (width, height) to pass to the matplotlib.pyplot.figure object
            points_size (float): The marker size

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the PCA plot


        """
        embeddings_array = np.array(list(self.embeddings.values()))
        element_array = np.array(self.element_list)

        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(embeddings_array)

        # Create a dataframe to store the dimension and the label for t-SNE transformation
        tsne_df = pd.DataFrame(
            {
                "tsne_dim1": tsne_result[:, 0],
                "tsne_dim2": tsne_result[:, 1],
                "element": element_array,
                "group": list(self.element_groups_dict.values()),
            }
        )
        # Create the t-SNE plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            x="tsne_dim1",
            y="tsne_dim2",
            data=tsne_df,
            hue=points_hue,
            s=points_size,
            ax=ax,
        )
        # lim = (tsne_result.min()-5, tsne_result.max()+5)
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        # Label the points
        for i in range(tsne_df.shape[0]):
            plt.text(
                x=tsne_df["tsne_dim1"][i],
                y=tsne_df["tsne_dim2"][i],
                s=tsne_df["element"][i],
            )

        return plt
