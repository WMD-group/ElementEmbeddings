"""
Provides the `Embedding` class.

This module enables the user load in elemetal representation data
and analyse it using statistical functions.

Typical usage example:
    megnet16 = Embedding.load_data('megnet16')
"""

import fnmatch
import json
import random
import warnings
from itertools import combinations_with_replacement
from os import path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
from pymatgen.core import Element
from scipy.stats import energy_distance, pearsonr, spearmanr, wasserstein_distance
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .utils.io import NumpyEncoder
from .utils.math import cosine_distance, cosine_similarity

module_directory = path.abspath(path.dirname(__file__))
data_directory = path.join(module_directory, "data")
pt_dir = path.join(data_directory, "element_data", "periodic-table-lookup-symbols.json")
with open(pt_dir) as f:
    pt = json.load(f)


class Embedding:
    """
    Represent an elemental representation.

    To load an embedding distributed from the package use the load_data() method.

    Works like a standard python dictionary. The keys are {element: vector} pairs.

    Adds a few convenience methods related to elemental representations.
    """

    def __init__(self, embeddings: dict, embedding_name: Optional[str] = None):
        """Initialise the Embedding class.

        Args:
            embeddings (dict): A {element_symbol: vector} dictionary
            embedding_name (str): The name of the elemental representation
        """
        self.embeddings = embeddings
        self.embedding_name = embedding_name

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
            self.dim: int = len(random.choice(list(self.embeddings.values())))
        else:
            self.dim: int = int(1)

        # Dummy initialisation for results
        self._data = []
        self._pca_data = None  # type: Optional[np.ndarray]
        self._tsne_data = None  # type: Optional[np.ndarray]
        self._umap_data = None  # type: Optional[np.ndarray]

    @staticmethod
    def load_data(embedding_name: Optional[str] = None):
        """
        Create an instance of the `Embedding` class from a default embedding file.

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
            embedding_name (str): The str_name of an embedding file.

        Returns:
            Embedding :class:`Embedding` instance.
        """
        _cbfv_files = {
            "magpie": "magpie.csv",
            "magpie_sc": "magpie_sc.json",
            "mat2vec": "mat2vec.csv",
            "matscholar": "matscholar-embedding.json",
            "megnet16": "megnet16.json",
            "mod_petti": "mod_petti.json",
            "oliynyk": "oliynyk.json",
            "oliynyk_sc": "oliynyk_sc.json",
            "random_200": "random_200_new.csv",
            "skipatom": "skipatom_20201009_induced.csv",
        }
        _cbfv_names = list(_cbfv_files.keys())
        _cbfv_names_others = [
            i
            for i in _cbfv_names
            if i not in ["skipatom", "random_200", "megnet16", "magpie", "mat2vec"]
        ]

        # Get the embeddings
        if embedding_name in _cbfv_files:
            if embedding_name in ["skipatom", "random_200", "magpie", "mat2vec"]:
                _csv = path.join(data_directory, _cbfv_files[embedding_name])
                df = pd.read_csv(_csv)
                # Convert df to a dictionary of (ele:embeddings) pairs
                elements = list(df["element"])
                df.drop(["element"], axis=1, inplace=True)
                embeds_array = df.to_numpy()
                embedding_data = {
                    elements[i]: embeds_array[i] for i in range(len(embeds_array))
                }

            elif embedding_name == "megnet16":
                megnet16_json = path.join(data_directory, _cbfv_files["megnet16"])
                with open(megnet16_json) as f:
                    embedding_data = json.load(f)
                # Remove 'Null' key from megnet embedding
                del embedding_data["Null"]

            elif embedding_name in _cbfv_names_others:
                _json = path.join(data_directory, _cbfv_files[embedding_name])
                with open(_json) as f:
                    embedding_data = json.load(f)

            # Load a json file from a file specified in the input
            else:
                with open(embedding_name) as f:
                    embedding_data = json.load(f)
        else:
            raise (
                ValueError(
                    f"{embedding_name} not in the data directory or not in directory."
                )
            )
        return Embedding(embedding_data, embedding_name)

    @staticmethod
    def from_json(embedding_json):
        """
        Create an instance of the Embedding class from a json file.

        Args:
            embedding_json (str): Filepath of the json file
        """
        # Need to add validation handling for JSONs in different formats
        with open(embedding_json) as f:
            embedding_data = json.load(f)
        return Embedding(embedding_data)

    @staticmethod
    def from_csv(embedding_csv):
        """
        Create an instance of the Embedding class from a csv file.

        The first column of the csv file must contain the elements and be named element.

        Args:
            embedding_csv (str): Filepath of the csv file
        """
        # Need to add validation handling for csv files
        df = pd.read_csv(embedding_csv)
        elements = list(df["element"])
        df.drop(["element"], axis=1, inplace=True)
        embeds_array = df.to_numpy()
        embedding_data = {
            elements[i]: embeds_array[i] for i in range(len(embeds_array))
        }
        return Embedding(embedding_data)

    def as_dataframe(self, columns: str = "components") -> pd.DataFrame:
        """
        Return the embedding as a pandas Dataframe.

        The first column is the elements and each other
        column represents a component of the embedding.

        Args:
            columns (str): A string to specify if the columns are the vector components
            and the index is the elements (`columns='components')
            or the columns are the elements (`columns='elements'`).

        Returns:
            df (pandas.DataFrame): A pandas dataframe object


        """
        embedding = self.embeddings
        df = pd.DataFrame(embedding)
        if columns == "components":
            return df.T
        elif columns == "elements":
            return df
        else:
            raise (
                ValueError(
                    f"{columns} is not a valid keyword argument. "
                    "Choose either 'components' or 'elements"
                )
            )

    def to(self, fmt: str = "", filename: Optional[str] = ""):
        """
        Output the embedding to a file.

        Args:
            fmt (str): The file format to output the embedding to.
            Options include "json" and "csv".
            filename (str): The name of the file to be outputted
        Returns:
            (str) if filename not specified, otherwise None.
        """
        fmt = fmt.lower()

        if fmt == "json" or fnmatch.fnmatch(filename, "*.json"):
            j = json.dumps(self.embeddings, cls=NumpyEncoder)
            if filename:
                if not filename.endswith(".json"):
                    filename = filename + ".json"
                with open(filename, "w") as file:
                    file.write(j)
            else:
                return j
        elif fmt == "csv" or fnmatch.fnmatch(filename, "*.csv"):
            if filename:
                if not filename.endswith(".csv"):
                    filename = filename + ".csv"
                self.as_dataframe().to_csv(filename, index_label="element")
            else:
                return self.as_dataframe().to_csv(index_label="element")

        else:
            raise ValueError(f"{str(fmt)} is an invalid file format")

    @property
    def element_list(self) -> list:
        """Return the elements of the embedding."""
        return list(self.embeddings.keys())

    def remove_elements(self, elements: Union[str, List[str]], inplace: bool = False):
        # TO-DO allow removal by atomic numbers
        """
        Remove elements from the Embedding instance.

        Args:
            elements (str,list(str)): An element symbol or a list of element symbols
            inplace (bool): If True, elements are removed from the Embedding instance.
            If false, the original embedding instance is unchanged
            and a new embedding instance with the elements removed is created.

        """
        if inplace:
            if isinstance(elements, str):
                del self.embeddings[elements]
            elif isinstance(elements, list):
                for el in elements:
                    del self.embeddings[el]
            return None
        else:
            embeddings_copy = self.embeddings.copy()
            if isinstance(elements, str):
                del embeddings_copy[elements]
            elif isinstance(elements, list):
                for el in elements:
                    del embeddings_copy[el]
            return Embedding(embeddings_copy, self.embedding_name)

    def citation(self) -> List[str]:
        """Return a citation for the embedding."""
        if self.embedding_name in ["magpie", "magpie_sc"]:
            citation = [
                "@article{ward2016general,"
                "title={A general-purpose machine learning framework for "
                "predicting properties of inorganic materials},"
                "author={Ward, Logan and Agrawal, Ankit and Choudhary, Alok "
                "and Wolverton, Christopher},"
                "journal={npj Computational Materials},"
                "volume={2},"
                "number={1},"
                "pages={1--7},"
                "year={2016},"
                "publisher={Nature Publishing Group}}"
            ]
        elif self.embedding_name == "mat2vec":
            citation = [
                "@article{tshitoyan2019unsupervised,"
                "title={Unsupervised word embeddings capture latent knowledge "
                "from materials science literature},"
                "author={Tshitoyan, Vahe and Dagdelen, John and Weston, Leigh "
                "and Dunn, Alexander and Rong, Ziqin and Kononova, Olga "
                "and Persson, Kristin A and Ceder, Gerbrand and Jain, Anubhav},"
                "journal={Nature},"
                "volume={571},"
                "number={7763},"
                "pages={95--98},"
                "year={2019},"
                "publisher={Nature Publishing Group} }"
            ]
        elif self.embedding_name == "matscholar":
            citation = [
                "@article{weston2019named,"
                "title={Named entity recognition and normalization applied to "
                "large-scale information extraction from the materials "
                "science literature},"
                "author={Weston, Leigh and Tshitoyan, Vahe and Dagdelen, John and "
                "Kononova, Olga and Trewartha, Amalie and Persson, Kristin A and "
                "Ceder, Gerbrand and Jain, Anubhav},"
                "journal={Journal of chemical information and modeling},"
                "volume={59},"
                "number={9},"
                "pages={3692--3702},"
                "year={2019},"
                "publisher={ACS Publications} }"
            ]

        elif self.embedding_name == "megnet16":
            citation = [
                "@article{chen2019graph,"
                "title={Graph networks as a universal machine learning framework "
                "for molecules and crystals},"
                "author={Chen, Chi and Ye, Weike and Zuo, Yunxing and "
                "Zheng, Chen and Ong, Shyue Ping},"
                "journal={Chemistry of Materials},"
                "volume={31},"
                "number={9},"
                "pages={3564--3572},"
                "year={2019},"
                "publisher={ACS Publications} }"
            ]

        elif self.embedding_name in ["oliynyk", "oliynyk_sc"]:
            citation = [
                "              @article{oliynyk2016high,"
                "title={High-throughput machine-learning-driven synthesis "
                "of full-Heusler compounds},"
                "author={Oliynyk, Anton O and Antono, Erin and Sparks, Taylor D and "
                "Ghadbeigi, Leila and Gaultois, Michael W and "
                "Meredig, Bryce and Mar, Arthur},"
                "journal={Chemistry of Materials},"
                "volume={28},"
                "number={20},"
                "pages={7324--7331},"
                "year={2016},"
                "publisher={ACS Publications} }"
            ]

        elif self.embedding_name == "skipatom":
            citation = [
                "@article{antunes2022distributed,"
                "title={Distributed representations of atoms and materials "
                "for machine learning},"
                "author={Antunes, Luis M and Grau-Crespo, Ricardo and Butler, Keith T},"
                "journal={npj Computational Materials},"
                "volume={8},"
                "number={1},"
                "pages={1--9},"
                "year={2022},"
                "publisher={Nature Publishing Group} }"
            ]
        elif self.embedding_name == "mod_petti":
            citation = [
                "@article{glawe2016optimal,"
                "title={The optimal one dimensional periodic table: "
                "a modified Pettifor chemical scale from data mining},"
                "author={Glawe, Henning and Sanna, Antonio and Gross, "
                "EKU and Marques, Miguel AL},"
                "journal={New Journal of Physics},"
                "volume={18},"
                "number={9},"
                "pages={093011},"
                "year={2016},"
                "publisher={IOP Publishing} }"
            ]

        else:
            citation = []

        return citation

    def _is_el_in_embedding(self, el: str) -> bool:
        """
        Check if an element is in the `Embedding` object.

        Args:
            el (str): An element symbol
        Returns:
            bool: True if el is in the Embedding, else False
        """
        if el in self.element_list:
            return True
        else:
            return False

    @property
    def element_groups_dict(self) -> Dict[str, str]:
        """
        Return a dictionary of {element: element type} pairs.

        e.g. {'He':'Noble gas'}

        """
        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[i] for i in self.element_list}

    def create_pairs(self):
        """Create all possible pairs of elements."""
        ele_list = self.element_list
        ele_pairs = combinations_with_replacement(ele_list, 2)
        return ele_pairs

    def stats_correlation_df(self) -> pd.DataFrame:
        """Return a pandas.DataFrame with correlation metrics.

         The columns of returned dataframe are:
        [element_1, element_2, pearson_corr, euclid_dist].
        """
        warnings.warn(
            "This method is deprecated and will be removed in a future release. ",
            DeprecationWarning,
        )
        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            pearson = pearsonr(self.embeddings[ele1], self.embeddings[ele2])
            dist = norm(self.embeddings[ele1] - self.embeddings[ele2])

            table.append((ele1, ele2, pearson[0], dist))
            if ele1 != ele2:
                table.append((ele2, ele1, pearson[0], dist))

        corr_df = pd.DataFrame(
            table,
            columns=[
                "ele_1",
                "ele_2",
                "pearson_corr",
                "euclid_dist",
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
                "pearson_corr",
            ]
        ]

        return corr_df

    def compute_correlation_metric(
        self, ele1: str, ele2: str, metric: str = "pearson"
    ) -> float:
        """
        Compute the correlation/similarity metric between two vectors.

        Allowed metrics:
        * Pearson
        * Spearman
        * Cosine similarity

        Args:
            ele1 (str): element symbol
            ele2 (str): element symbol
            metric (str): name of a correlation metric.
            Options are "spearman", "pearson" and "cosine_similarity".

        Returns:
            float: correlation/similarity metric
        """
        # Define the allowable metrics
        scipy_corrs = {"pearson": pearsonr, "spearman": spearmanr}

        if metric == "pearson":
            return scipy_corrs[metric](
                self.embeddings[ele1], self.embeddings[ele2]
            ).statistic
        elif metric == "spearman":
            return scipy_corrs[metric](
                self.embeddings[ele1], self.embeddings[ele2]
            ).correlation
        elif metric == "cosine_similarity":
            return cosine_similarity(self.embeddings[ele1], self.embeddings[ele2])

    def compute_distance_metric(
        self, ele1: str, ele2: str, metric: str = "euclidean"
    ) -> float:
        """
        Compute distance metric between two vectors.

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy
        * cosine_distance

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

        valid_metrics = scikit_metrics + list(scipy_metrics.keys()) + ["cosine"]

        # Validate if the elements are within the embedding vector
        if not all([self._is_el_in_embedding(ele1), self._is_el_in_embedding(ele2)]):
            if not self._is_el_in_embedding(ele1):
                print(f"{ele1} is not an element included within the atomic embeddings")
                raise ValueError

            elif not self._is_el_in_embedding(ele2):
                print(f"{ele2} is not an element included within the atomic embeddings")
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
        elif metric == "cosine_distance":
            return cosine_distance(self.embeddings[ele1], self.embeddings[ele2])

        else:
            print(
                "Invalid distance metric."
                f"Use one of the following metrics:{valid_metrics}"
            )
            raise ValueError

    def pearson_pivot_table(self) -> pd.DataFrame:
        """
        Return a pandas.DataFrame style pivot object.

        The index and column are the mendeleev number of the element pairs
        and the values being the pearson correlation metrics.
        """
        warnings.warn(
            "This method is deprecated and will be removed in a future release. ",
            DeprecationWarning,
        )
        corr_df = self.correlation_df()
        pearson_pivot = corr_df.pivot_table(
            values="pearson_corr", index="mend_1", columns="mend_2"
        )
        return pearson_pivot

    def distance_df(self, metric: str = "euclidean") -> pd.DataFrame:
        """
        Return a dataframe with columns ["ele_1", "ele_2", metric].

        Allowed metrics:

        * euclidean
        * manhattan
        * chebyshev
        * wasserstein
        * energy

        Args:
            metric (str): A distance metric.

        Returns:
            df (pandas.DataFrame): A dataframe with columns ["ele_1", "ele_2", metric].
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

        Z_1 = [(pt[ele]["number"], ele) for ele in corr_df["ele_1"]]
        Z_2 = [(pt[ele]["number"], ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df["Z_1"] = Z_1
        corr_df["Z_2"] = Z_2

        corr_df = corr_df[["ele_1", "ele_2", "mend_1", "mend_2", "Z_1", "Z_2", metric]]

        return corr_df

    def correlation_df(self, metric: str = "pearson") -> pd.DataFrame:
        """
        Return a dataframe with columns ["ele_1", "ele_2", metric].

        Allowed metrics:

        * pearson
        * spearman
        * cosine_similarity


        Args:
            metric (str): A distance metric.

        Returns:
            df (pandas.DataFrame): A dataframe with columns ["ele_1", "ele_2", metric].
        """
        ele_pairs = self.create_pairs()
        table = []
        for ele1, ele2 in ele_pairs:
            dist = self.compute_correlation_metric(ele1, ele2, metric=metric)
            table.append((ele1, ele2, dist))
            if ele1 != ele2:
                table.append((ele2, ele1, dist))
        corr_df = pd.DataFrame(table, columns=["ele_1", "ele_2", metric])

        mend_1 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_1"]]
        mend_2 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_2"]]

        Z_1 = [(pt[ele]["number"], ele) for ele in corr_df["ele_1"]]
        Z_2 = [(pt[ele]["number"], ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df["Z_1"] = Z_1
        corr_df["Z_2"] = Z_2

        corr_df = corr_df[["ele_1", "ele_2", "mend_1", "mend_2", "Z_1", "Z_2", metric]]

        return corr_df

    def distance_pivot_table(
        self, metric: str = "euclidean", sortby: str = "mendeleev"
    ) -> pd.DataFrame:
        """
        Return a pandas.DataFrame style pivot.

        The index and column being either the mendeleev number or atomic number
        of the element pairs and the values being a user-specified distance metric.

        Args:
            metric (str): A distance metric.
            sortby (str): Sort the pivot table by either "mendeleev" or "atomic_number".

        Returns:
            distance_pivot (pandas.DataFrame): A pandas DataFrame pivot table.
        """
        corr_df = self.distance_df(metric=metric)
        if sortby == "mendeleev":
            distance_pivot = corr_df.pivot_table(
                values=metric, index="mend_1", columns="mend_2"
            )
            return distance_pivot
        elif sortby == "atomic_number":
            distance_pivot = corr_df.pivot_table(
                values=metric, index="Z_1", columns="Z_2"
            )
            return distance_pivot

    def correlation_pivot_table(
        self, metric: str = "pearson", sortby: str = "mendeleev"
    ) -> pd.DataFrame:
        """
        Return a pandas.DataFrame style pivot.

        The index and column being either the mendeleev number or atomic number
        of the element pairs and the values being a user-specified distance metric.

        Args:
            metric (str): A distance metric.
            sortby (str): Sort the pivot table by either "mendeleev" or "atomic_number".

        Returns:
            distance_pivot (pandas.DataFrame): A pandas DataFrame pivot table.
        """
        corr_df = self.correlation_df(metric=metric)
        if sortby == "mendeleev":
            correlation_pivot = corr_df.pivot_table(
                values=metric, index="mend_1", columns="mend_2"
            )
            return correlation_pivot
        elif sortby == "atomic_number":
            correlation_pivot = corr_df.pivot_table(
                values=metric, index="Z_1", columns="Z_2"
            )
            return correlation_pivot

    def plot_pearson_correlation(self, figsize: Tuple[int, int] = (24, 24), **kwargs):
        """
        Plot the heatmap of the pearson correlation values.

        Args:
            figsize (tuple): A tuple of (width, height).
            **kwargs: Other keyword arguments to be passed to sns.heatmap

        Returns:
            ax (matplotlib Axes): An Axes object with the heatmap

        """
        warnings.warn(
            "This method is deprecated and will be removed in a future release. ",
            DeprecationWarning,
        )
        pearson_pivot = self.pearson_pivot_table()

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            pearson_pivot, cmap="bwr", square=True, linecolor="k", **kwargs
        )

        return ax

    def plot_distance_correlation(
        self, metric: str = "euclidean", figsize: Tuple[int, int] = (24, 24), **kwargs
    ):
        """
        Plot the heatmap of the pairwise distance metrics.

        Args:
            metric (str): A valid distance metric
            figsize (tuple): A tuple of (width, height)

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the heatmap

        """
        warnings.warn(
            "This method is deprecated and will be removed in a future release. ",
            DeprecationWarning,
        )
        distance_pivot = self.distance_pivot_table(metric=metric)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            distance_pivot, cmap="bwr", square=True, linecolor="k", **kwargs
        )

        return ax

    def calculate_PC(self, n_components: int = 2, scale: bool = True, **kwargs):
        """Calculate the principal componenets (PC) of the embeddings."""
        if scale:
            embeddings_array = StandardScaler().fit_transform(
                np.array(list(self.embeddings.values()))
            )
        else:
            embeddings_array = np.array(list(self.embeddings.values()))

        pca = decomposition.PCA(
            n_components=n_components, **kwargs
        )  # project to N dimensions
        pca.fit(embeddings_array)
        self._pca_data = pca.transform(embeddings_array)
        return self._pca_data

    def calculate_tSNE(self, n_components: int = 2, scale: bool = True, **kwargs):
        """Calculate t-SNE components."""
        if scale:
            embeddings_array = StandardScaler().fit_transform(
                np.array(list(self.embeddings.values()))
            )
        else:
            embeddings_array = np.array(list(self.embeddings.values()))

        tsne = TSNE(n_components=n_components, **kwargs)
        tsne_result = tsne.fit_transform(embeddings_array)
        self._tsne_data = tsne_result
        return self._tsne_data

    def calculate_UMAP(self, n_components: int = 2, scale: bool = True, **kwargs):
        """Calculate UMAP embeddings."""
        if scale:
            embeddings_array = StandardScaler().fit_transform(
                np.array(list(self.embeddings.values()))
            )
        else:
            embeddings_array = np.array(list(self.embeddings.values()))

        umap = UMAP(n_components=n_components, **kwargs)
        umap_result = umap.fit_transform(embeddings_array)
        self._umap_data = umap_result
        return self._umap_data

    def plot_PCA_2D(
        self,
        figsize: Tuple[int, int] = (16, 12),
        points_hue: str = "group",
        points_size: int = 200,
        **kwargs,
    ):
        """Plot a PCA plot of the atomic embedding.

        Args:
            figsize (tuple): A tuple of (width, height)
            points_size (float): The marker size

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the PCA plot

        """
        embeddings_array = np.array(list(self.embeddings.values()))
        element_array = np.array(self.element_list)

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
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            x="pca_dim1",
            y="pca_dim2",
            data=pca_df,
            hue=points_hue,
            s=points_size,
            **kwargs,
            ax=ax,
        )

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        for i in range(len(X)):
            plt.text(x=pca_dim1[i], y=pca_dim2[i], s=element_array[i])

        return plt

    def plot_tSNE(
        self,
        n_components: str = 2,
        figsize: Tuple[int, int] = (16, 12),
        points_hue: str = "group",
        points_size: int = 200,
        **kwargs,
    ):
        """Plot a t-SNE plot of the atomic embedding.

        Args:
            n_components (int): Number of t-SNE components to plot.
            figsize (tuple): A tuple of (width, height)
            points_size (float): The marker size

        Returns:
            ax (matplotlib.axes.Axes): An Axes object with the PCA plot


        """
        embeddings_array = np.array(list(self.embeddings.values()))
        element_array = np.array(self.element_list)

        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(embeddings_array)

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
