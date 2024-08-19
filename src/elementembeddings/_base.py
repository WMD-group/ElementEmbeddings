from __future__ import annotations

import json
import random
import warnings
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from os import path

import numpy as np
import pandas as pd
from openTSNE import TSNE
from pymatgen.core import Element
from scipy.stats import energy_distance, pearsonr, spearmanr, wasserstein_distance
from sklearn import decomposition
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .utils.config import CITATIONS
from .utils.math import cosine_distance, cosine_similarity
from .utils.species import parse_species

module_directory = path.abspath(path.dirname(__file__))
data_directory = path.join(module_directory, "data")
pt_dir = path.join(data_directory, "element_data", "periodic-table-lookup-symbols.json")
with open(pt_dir) as f:
    pt = json.load(f)


class EmbeddingBase(ABC):
    """Base class for embeddings."""

    def __init__(
        self,
        embeddings: dict,
        embedding_name: str | None = None,
        feature_labels: list[str] | None = None,
    ) -> None:
        """Initialise the embedding base class.

        Args:
        ----
            embeddings (dict): A dictionary of embeddings.
            embedding_name (str): The name of the embedding.
            feature_labels (list): A list of feature labels.
        """
        self.embeddings = embeddings
        self.embedding_name = embedding_name

        self.feature_labels = feature_labels
        if self._is_standardised():
            self.is_standardised = True
        else:
            self.is_standardised = False

        # Grab a random value from the embedding vector
        _rand_embed = random.choice(list(self.embeddings.values()))
        # Convert embeddings to numpy array if not already a numpy array
        if not isinstance(_rand_embed, np.ndarray):
            self.embeddings = {ele: np.array(self.embeddings[ele]) for ele in self.embeddings}

        # Determines if the embedding vector has a length attribute
        # (i.e. is not a scalar int or float)
        # If the 'vector' is a scalar/float, the representation is linear
        # A linear representation gets converted to a one-hot vector
        if hasattr(_rand_embed, "__len__") and (not isinstance(_rand_embed, str)):
            self.embedding_type: str = "vector"
            self.dim: int = len(random.choice(list(self.embeddings.values())))
        else:
            self.embedding_type: str = "one-hot"

            # If the embedding is a one-hot vector,
            # the dimension is the number of elements/species
            sorted_embedding = sorted(self.embeddings.items(), key=lambda x: x[1])
            elements = np.loadtxt(
                f"{data_directory}/element_data/ordered_periodic.txt",
                dtype=str,
            )
            # Exceptions for mod_petti
            if self.embedding_name == "mod_petti":
                sorted_embedding = {el: num for el, num in sorted_embedding if el in elements[:103]}
            else:
                sorted_embedding = {el: num for el, num in sorted_embedding if el in elements[:118]}
            self.feature_labels = list(sorted_embedding.keys())
            self.embeddings = {}
            for el, num in sorted_embedding.items():
                self.embeddings[el] = np.zeros(len(sorted_embedding))
                self.embeddings[el][num] = 1
            self.dim = len(random.choice(list(self.embeddings.values())))

        # If feature labels are not provided, use the range of the embedding dimension
        if not self.feature_labels:
            self.feature_labels = list(range(self.dim))

    @staticmethod
    @abstractmethod
    def load_data(embedding_name: str):
        """Abstract method for loading data."""

    @staticmethod
    @abstractmethod
    def from_csv(csv_path: str):
        """Abstract method for loading data from a csv."""

    @staticmethod
    def from_json(json_path: str, embedding_name: str | None = None):
        """Create an embedding from a json file.

        Args:
        ----
            json_path (str): The path to the json file.
            embedding_name (str): The name of the embedding.
        """
        with open(json_path) as f:
            embeddings = json.load(f)
        return EmbeddingBase(embeddings, embedding_name)

    def citation(self):
        """Return the citation for the embedding."""
        try:
            return CITATIONS[self.embedding_name]
        except KeyError:
            return None

    def _is_standardised(self):
        """Check if the embedding is standardised.

        Mean must be 0 and standard deviation must be 1.
        """
        return np.isclose(
            np.mean(np.array(list(self.embeddings.values()))),
            0,
        ) and np.isclose(np.std(np.array(list(self.embeddings.values()))), 1)

    def _is_el_sp_in_embedding(self, el_sp: str):
        """Check if an element/species is in the embedding."""
        return el_sp in self.embeddings

    def _embeddings_keys_list(self):
        """Return the keys of the embedding as a list."""
        return list(self.embeddings.keys())

    def create_pairs(self):
        """Create all possible pairs of elements/species."""
        return combinations_with_replacement(self._embeddings_keys_list(), 2)

    def standardise(self, inplace: bool = False):
        """Standardise the embedding.

        Mean is 0 and standard deviation is 1.

        Args:
        ----
            inplace (bool): Whether to change the embedding in place.

        Returns:
        -------
            None if inplace is True, otherwise returns the standardised embedding.
        """
        if self._is_standardised():
            warnings.warn(
                "Embedding is already standardised." "Returning None and not changing the embedding",
            )
            return None
        else:
            embeddings_copy = self.embeddings.copy()
            embeddings_array = np.array(list(embeddings_copy.values()))
            embeddings_array = StandardScaler().fit_transform(embeddings_array)
            for el, emb in zip(embeddings_copy.keys(), embeddings_array):
                embeddings_copy[el] = emb
            if inplace:
                self.embeddings = embeddings_copy
                self.is_standardised = True
                return None
            else:
                return EmbeddingBase(embeddings_copy, self.embedding_name)

    def calculate_pca(self, n_components: int = 2, standardise: bool = True, **kwargs):
        """Calculate the principal components (PC) of the embeddings.

        Args:
        ----
            n_components (int): The number of components to project the embeddings to.
            standardise (bool): Whether to standardise the embeddings before projecting.
            **kwargs: Other keyword arguments to be passed to PCA.
        """
        if standardise:
            if self.is_standardised:
                embeddings_array = np.array(list(self.embeddings.values()))
            else:
                self.standardise(inplace=True)
                embeddings_array = np.array(list(self.embeddings.values()))
        else:
            warnings.warn(
                """It is recommended to scale the embeddings
                before projecting with PCA.
                To do so, set `standardise=True`.""",
            )
            embeddings_array = np.array(list(self.embeddings.values()))

        pca = decomposition.PCA(
            n_components=n_components,
            **kwargs,
        )  # project to N dimensions
        pca.fit(embeddings_array)
        return pca.transform(embeddings_array)

    def calculate_tsne(self, n_components: int = 2, standardise: bool = True, **kwargs):
        """Calculate t-SNE components.

        Args:
        ----
            n_components (int): The number of components to project the embeddings to.
            standardise (bool): Whether to standardise the embeddings before projecting.
            **kwargs: Other keyword arguments to be passed to t-SNE.
        """
        if standardise:
            if self.is_standardised:
                embeddings_array = np.array(list(self.embeddings.values()))
            else:
                self.standardise(inplace=True)
                embeddings_array = np.array(list(self.embeddings.values()))
        else:
            warnings.warn(
                """It is recommended to scale the embeddings
                before projecting with t-SNE.
                To do so, set `standardise=True`.""",
            )
            embeddings_array = np.array(list(self.embeddings.values()))

        tsne = TSNE(n_components=n_components, **kwargs)
        return tsne.fit(embeddings_array)

    def calculate_umap(self, n_components: int = 2, standardise: bool = True, **kwargs):
        """Calculate UMAP embeddings.

        Args:
        ----
            n_components (int): The number of components to project the embeddings to.
            standardise (bool): Whether to scale the embeddings before projecting.
            **kwargs: Other keyword arguments to be passed to UMAP.
        """
        if standardise:
            if self.is_standardised:
                embeddings_array = np.array(list(self.embeddings.values()))
            else:
                self.standardise(inplace=True)
                embeddings_array = np.array(list(self.embeddings.values()))
        else:
            warnings.warn(
                """It is recommended to scale the embeddings
                before projecting with UMAP.
                To do so, set `standardise=True`.""",
            )
            embeddings_array = np.array(list(self.embeddings.values()))

        umap = UMAP(n_components=n_components, **kwargs)
        return umap.fit_transform(embeddings_array)

    def compute_correlation_metric(
        self,
        el_sp1: str,
        el_sp2: str,
        metric: str = "pearson",
    ) -> float:
        """Compute the correlation/similarity metric between two elements/species.

        Allowed metrics:
        * Pearson
        * Spearman
        * Cosine similarity

        Args:
        ----
            el_sp1 (str): element symbol
            el_sp2 (str): element symbol
            metric (str): name of a correlation metric.
                Options are "spearman", "pearson" and "cosine_similarity".

        Returns:
        -------
            float: correlation/similarity metric
        """
        # Validate if the elements are within the embedding vector
        if not all([self._is_el_sp_in_embedding(el_sp1), self._is_el_sp_in_embedding(el_sp2)]):
            if not self._is_el_sp_in_embedding(el_sp1):
                print(f"{el_sp1} is not an element/species included within the embeddings")
                raise ValueError

            if not self._is_el_sp_in_embedding(el_sp2):
                print(f"{el_sp2} is not an element/species included within the embeddings")
                raise ValueError
        if metric == "pearson":
            return pearsonr(self.embeddings[el_sp1], self.embeddings[el_sp2])[0]
        elif metric == "spearman":
            return spearmanr(self.embeddings[el_sp1], self.embeddings[el_sp2])[0]
        elif metric == "cosine_similarity":
            return cosine_similarity(self.embeddings[el_sp1], self.embeddings[el_sp2])
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_distance_metric(
        self,
        el_sp1: str,
        el_sp2: str,
        metric: str = "euclidean",
    ) -> float:
        """Compute distance metric between two vectors.

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy
        * cosine_distance

        Args:
        ----
            el_sp1 (str): element symbol
            el_sp2 (str): element symbol
            metric (str): name of a distance metric

        Returns:
        -------
            distance (float): distance between embedding vectors
        """
        # Define the allowable metrics
        scikit_metrics = ["euclidean", "manhattan", "chebyshev"]

        scipy_metrics = {"wasserstein": wasserstein_distance, "energy": energy_distance}

        valid_metrics = scikit_metrics + list(scipy_metrics.keys()) + ["cosine"]

        # Validate if the elements are within the embedding vector
        if not all([self._is_el_sp_in_embedding(el_sp1), self._is_el_sp_in_embedding(el_sp2)]):
            if not self._is_el_sp_in_embedding(el_sp1):
                print(f"{el_sp1} is not an element/species included within the embeddings")
                raise ValueError

            if not self._is_el_sp_in_embedding(el_sp2):
                print(f"{el_sp2} is not an element/species included within the embeddings")
                raise ValueError

        # Compute the distance measure
        if metric in scikit_metrics:
            distance = DistanceMetric.get_metric(metric)

            return distance.pairwise(
                self.embeddings[el_sp1].reshape(1, -1),
                self.embeddings[el_sp2].reshape(1, -1),
            )[0][0]

        elif metric in scipy_metrics:
            return scipy_metrics[metric](self.embeddings[el_sp1], self.embeddings[el_sp2])
        elif metric == "cosine_distance":
            return cosine_distance(self.embeddings[el_sp1], self.embeddings[el_sp2])

        else:
            print(
                "Invalid distance metric." f"Use one of the following metrics:{valid_metrics}",
            )
            raise ValueError

    def distance_df(self, metric: str = "euclidean") -> pd.DataFrame:
        """Return a dataframe with columns ["ele_1", "ele_2", metric].

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy

        Args:
        ----
            metric (str): A distance metric.

        Returns:
        -------
            df (pandas.DataFrame): A dataframe with columns ["ele_1", "ele_2", metric].
        """
        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            dist = self.compute_distance_metric(ele1, ele2, metric=metric)
            table.append((ele1, ele2, dist))
            if ele1 != ele2:
                table.append((ele2, ele1, dist))
        dist_df = pd.DataFrame(table, columns=["ele_1", "ele_2", metric])

        mend_1 = [(Element(parse_species(ele)[0]).mendeleev_no, ele) for ele in dist_df["ele_1"]]
        mend_2 = [(Element(parse_species(ele)[0]).mendeleev_no, ele) for ele in dist_df["ele_2"]]

        Z_1 = [(pt[parse_species(ele)[0]]["number"], ele) for ele in dist_df["ele_1"]]
        Z_2 = [(pt[parse_species(ele)[0]]["number"], ele) for ele in dist_df["ele_2"]]

        dist_df["mend_1"] = mend_1
        dist_df["mend_2"] = mend_2

        dist_df["Z_1"] = Z_1
        dist_df["Z_2"] = Z_2

        return dist_df[["ele_1", "ele_2", "mend_1", "mend_2", "Z_1", "Z_2", metric]]

    def correlation_df(self, metric: str = "pearson") -> pd.DataFrame:
        """Return a dataframe with columns ["ele_1", "ele_2", metric].

        Allowed metrics:

        * pearson
        * spearman
        * cosine_similarity


        Args:
        ----
            metric (str): A correlation/similarity metric.

        Returns:
        -------
            df (pandas.DataFrame): A dataframe with columns ["ele_1", "ele_2", metric].
        """
        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            corr = self.compute_correlation_metric(ele1, ele2, metric=metric)
            table.append((ele1, ele2, corr))
            if ele1 != ele2:
                table.append((ele2, ele1, corr))
        corr_df = pd.DataFrame(table, columns=["ele_1", "ele_2", metric])

        mend_1 = [(Element(parse_species(ele)[0]).mendeleev_no, ele) for ele in corr_df["ele_1"]]
        mend_2 = [(Element(parse_species(ele)[0]).mendeleev_no, ele) for ele in corr_df["ele_2"]]

        Z_1 = [(pt[parse_species(ele)[0]]["number"], ele) for ele in corr_df["ele_1"]]
        Z_2 = [(pt[parse_species(ele)[0]]["number"], ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df["Z_1"] = Z_1
        corr_df["Z_2"] = Z_2

        return corr_df[["ele_1", "ele_2", "mend_1", "mend_2", "Z_1", "Z_2", metric]]

    def distance_pivot_table(
        self,
        metric: str = "euclidean",
        sortby: str = "mendeleev",
    ) -> pd.DataFrame:
        """Return a pandas.DataFrame style pivot.

        The index and column being either the mendeleev number or atomic number
        of the element pairs and the values being a user-specified distance metric.

        Args:
        ----
            metric (str): A distance metric.
            sortby (str): Sort the pivot table by either "mendeleev" or "atomic_number".

        Returns:
        -------
            distance_pivot (pandas.DataFrame): A pandas DataFrame pivot table.
        """
        corr_df = self.distance_df(metric=metric)
        if sortby == "mendeleev":
            return corr_df.pivot_table(
                values=metric,
                index="mend_1",
                columns="mend_2",
            )
        elif sortby == "atomic_number":
            return corr_df.pivot_table(
                values=metric,
                index="Z_1",
                columns="Z_2",
            )
        return None

    def correlation_pivot_table(
        self,
        metric: str = "pearson",
        sortby: str = "mendeleev",
    ) -> pd.DataFrame:
        """Return a pandas.DataFrame style pivot.

        The index and column being either the mendeleev number or atomic number
        of the element pairs and the values being a user-specified distance metric.

        Args:
        ----
            metric (str): A correlation/similarity metric.
            sortby (str): Sort the pivot table by either "mendeleev" or "atomic_number".

        Returns:
        -------
            correlation_pivot (pandas.DataFrame): A pandas DataFrame pivot table.
        """
        corr_df = self.correlation_df(metric=metric)
        if sortby == "mendeleev":
            return corr_df.pivot_table(
                values=metric,
                index="mend_1",
                columns="mend_2",
            )
        elif sortby == "atomic_number":
            return corr_df.pivot_table(
                values=metric,
                index="Z_1",
                columns="Z_2",
            )
        return None
