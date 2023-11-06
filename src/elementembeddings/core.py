"""Provides the `Embedding` class.

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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pymatgen.core import Element
from sklearn.preprocessing import StandardScaler

from ._base import EmbeddingBase
from .utils.config import CITATIONS, DEFAULT_ELEMENT_EMBEDDINGS
from .utils.io import NumpyEncoder
from .utils.species import parse_species

module_directory = path.abspath(path.dirname(__file__))
data_directory = path.join(module_directory, "data")
pt_dir = path.join(data_directory, "element_data", "periodic-table-lookup-symbols.json")
with open(pt_dir) as f:
    pt = json.load(f)


class Embedding(EmbeddingBase):
    """Represent an elemental representation.

    To load an embedding distributed from the package use the load_data() method.

    Works like a standard python dictionary. The keys are {element: vector} pairs.

    Adds a few convenience methods related to elemental representations.
    """

    def __init__(
        self,
        embeddings: dict,
        embedding_name: Optional[str] = None,
        feature_labels: Optional[List[str]] = None,
    ) -> None:
        """Initialise the Embedding class.

        Args:
        ----
            embeddings (dict): A {element_symbol: vector} dictionary
            embedding_name (str): The name of the elemental representation
            feature_labels (list(str)): A list of feature labels
        """
        self.embeddings = embeddings
        self.embedding_name = embedding_name
        self.feature_labels = feature_labels
        if not self._is_standardised():
            self.is_standardised = False
        else:
            self.is_standardised = True

        # Grab a random value from the embedding vector
        _rand_embed = random.choice(list(self.embeddings.values()))
        # Convert embeddings to numpy array if not already a numpy array
        if not isinstance(_rand_embed, np.ndarray):
            self.embeddings = {
                ele: np.array(self.embeddings[ele]) for ele in self.embeddings
            }

        # Determines if the embedding vector has a length attribute
        # (i.e. is not a scalar int or float)
        # If the 'vector' is a scalar/float, the representation is linear
        # A linear representation gets converted to a one-hot vector
        if hasattr(_rand_embed, "__len__") and (not isinstance(_rand_embed, str)):
            self.embedding_type: str = "vector"
            self.dim: int = len(random.choice(list(self.embeddings.values())))
        else:
            self.embedding_type: str = "linear"

        # Create one-hot vectors for a scalar representation
        if self.embedding_type == "linear":
            sorted_embedding = sorted(self.embeddings.items(), key=lambda x: x[1])
            elements = np.loadtxt(
                f"{data_directory}/element_data/ordered_periodic.txt",
                dtype=str,
            )
            if self.embedding_name == "mod_petti":
                sorted_embedding = {
                    el: num for el, num in sorted_embedding if el in elements[:103]
                }
            else:
                sorted_embedding = {
                    el: num for el, num in sorted_embedding if el in elements[:118]
                }
            self.feature_labels = list(sorted_embedding.keys())
            self.embeddings = {}

            for el, num in sorted_embedding.items():
                self.embeddings[el] = np.zeros(len(sorted_embedding))
                self.embeddings[el][num] = 1
            self.dim = len(random.choice(list(self.embeddings.values())))

        if not self.feature_labels:
            self.feature_labels = list(range(self.dim))
        else:
            self.feature_labels = self.feature_labels

        # Dummy initialisation for results
        self._data = []
        self._pca_data = None  # type: Optional[np.ndarray]
        self._tsne_data = None  # type: Optional[np.ndarray]
        self._umap_data = None  # type: Optional[np.ndarray]

    @staticmethod
    def load_data(embedding_name: Optional[str] = None):
        """Create an instance of the `Embedding` class from a default embedding file.

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
        | Atomic Number           | atomic       |


        Args:
        ----
            embedding_name (str): The str_name of an embedding file.

        Returns:
        -------
            Embedding :class:`Embedding` instance.
        """
        if DEFAULT_ELEMENT_EMBEDDINGS[embedding_name].endswith(".csv"):
            return Embedding.from_csv(
                path.join(
                    data_directory,
                    "element_representations",
                    DEFAULT_ELEMENT_EMBEDDINGS[embedding_name],
                ),
                embedding_name,
            )
        elif "megnet" in DEFAULT_ELEMENT_EMBEDDINGS[embedding_name]:
            return Embedding.from_json(
                path.join(
                    data_directory,
                    "element_representations",
                    DEFAULT_ELEMENT_EMBEDDINGS[embedding_name],
                ),
                embedding_name,
            ).remove_elements(["Null"])
        elif DEFAULT_ELEMENT_EMBEDDINGS[embedding_name].endswith(".json"):
            return Embedding.from_json(
                path.join(
                    data_directory,
                    "element_representations",
                    DEFAULT_ELEMENT_EMBEDDINGS[embedding_name],
                ),
                embedding_name,
            )
        else:
            return None

    @staticmethod
    def from_json(embedding_json, embedding_name: Optional[str] = None):
        """Create an instance of the Embedding class from a json file.

        Args:
        ----
            embedding_json (str): Filepath of the json file
            embedding_name (str): The name of the elemental representation
        """
        # Need to add validation handling for JSONs in different formats
        with open(embedding_json) as f:
            embedding_data = json.load(f)
        return Embedding(embedding_data, embedding_name)

    @staticmethod
    def from_csv(embedding_csv, embedding_name: Optional[str] = None):
        """Create an instance of the Embedding class from a csv file.

        The first column of the csv file must contain the elements and be named element.

        Args:
        ----
            embedding_csv (str): Filepath of the csv file
            embedding_name (str): The name of the elemental representation

        """
        # Need to add validation handling for csv files
        df = pd.read_csv(embedding_csv)
        elements = list(df["element"])
        df = df.drop(["element"], axis=1)
        feature_labels = list(df.columns)
        embeds_array = df.to_numpy()
        embedding_data = {
            elements[i]: embeds_array[i] for i in range(len(embeds_array))
        }
        return Embedding(embedding_data, embedding_name, feature_labels)

    def as_dataframe(self, columns: str = "components") -> pd.DataFrame:
        """Return the embedding as a pandas Dataframe.

        The first column is the elements and each other
        column represents a component of the embedding.

        Args:
        ----
            columns (str): A string to specify if the columns are the vector components
            and the index is the elements (`columns='components'`)
            or the columns are the elements (`columns='elements'`).

        Returns:
        -------
            df (pandas.DataFrame): A pandas dataframe object


        """
        embedding = self.embeddings
        df = pd.DataFrame(embedding, index=self.feature_labels)
        if columns == "components":
            return df.T
        elif columns == "elements":
            return df
        else:
            msg = (
                f"{columns} is not a valid keyword argument. "
                f"Choose either 'components' or 'elements"
            )
            raise (
                ValueError(
                    msg,
                )
            )

    def to(self, fmt: str = "", filename: Optional[str] = ""):
        """Output the embedding to a file.

        Args:
        ----
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
                    return None
            else:
                return j
        elif fmt == "csv" or fnmatch.fnmatch(filename, "*.csv"):
            if filename:
                if not filename.endswith(".csv"):
                    filename = filename + ".csv"
                self.as_dataframe().to_csv(filename, index_label="element")
                return None
            else:
                return self.as_dataframe().to_csv(index_label="element")

        else:
            msg = f"{fmt!s} is an invalid file format"
            raise ValueError(msg)

    @property
    def element_list(self) -> list:
        """Return the elements of the embedding."""
        return self._embeddings_keys_list()

    def remove_elements(self, elements: Union[str, List[str]], inplace: bool = False):
        # TO-DO allow removal by atomic numbers
        """Remove elements from the Embedding instance.

        Args:
        ----
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

    def standardise(self, inplace: bool = False):
        """Standardise the embeddings.

        Mean is 0 and standard deviation is 1.

        """
        if self._is_standardised():
            warnings.warn(
                "Embedding is already standardised. "
                "Returning None and not changing the embedding.",
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
                return Embedding(embeddings_copy, self.embedding_name)

    def citation(self) -> List[str]:
        """Return a citation for the embedding."""
        try:
            citation = CITATIONS[self.embedding_name]
        except KeyError:
            citation = None
        return citation

    @property
    def element_groups_dict(self) -> Dict[str, str]:
        """Return a dictionary of {element: element type} pairs.

        e.g. {'He':'Noble gas'}

        """
        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[i] for i in self.element_list}

    def create_pairs(self):
        """Create all possible pairs of elements."""
        ele_list = self.element_list
        return combinations_with_replacement(ele_list, 2)

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
        corr_df = pd.DataFrame(table, columns=["ele_1", "ele_2", metric])

        mend_1 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_1"]]
        mend_2 = [(Element(ele).mendeleev_no, ele) for ele in corr_df["ele_2"]]

        Z_1 = [(pt[ele]["number"], ele) for ele in corr_df["ele_1"]]
        Z_2 = [(pt[ele]["number"], ele) for ele in corr_df["ele_2"]]

        corr_df["mend_1"] = mend_1
        corr_df["mend_2"] = mend_2

        corr_df["Z_1"] = Z_1
        corr_df["Z_2"] = Z_2

        return corr_df[["ele_1", "ele_2", "mend_1", "mend_2", "Z_1", "Z_2", metric]]

    def correlation_df(self, metric: str = "pearson") -> pd.DataFrame:
        """Return a dataframe with columns ["ele_1", "ele_2", metric].

        Allowed metrics:

        * pearson
        * spearman
        * cosine_similarity


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
            metric (str): A distance metric.
            sortby (str): Sort the pivot table by either "mendeleev" or "atomic_number".

        Returns:
        -------
            distance_pivot (pandas.DataFrame): A pandas DataFrame pivot table.
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


class SpeciesEmbedding(EmbeddingBase):
    """Represent an ion representation.

    To load an embedding distributed from the package use the load_data() method.

    Works like a standard python dictionary. The keys are {species: vector} pairs.
    """

    def __init__(
        self,
        embeddings: dict,
        embedding_name: Optional[str] = None,
        feature_labels: Optional[List[str]] = None,
    ) -> None:
        """Create an instance of the SpeciesEmbedding class.

        Args:
        ----
            embeddings (dict): A dictionary of {species: vector} pairs.
            embedding_name (str): The name of the species representation
            feature_labels (list(str)): A list of feature labels for the embedding

        """
        self.embeddings = embeddings
        self.embedding_name = embedding_name
        self.feature_labels = feature_labels

    @staticmethod
    def load_data(embedding_name: str):
        """Create a `SpeciesEmbedding` from a preset embedding file.

        Args:
        ----
            embedding_name (str): The name of the species representation

        Returns:
        -------
            SpeciesEmbedding :class:`SpeciesEmbedding` instance.
        """
        raise NotImplementedError

    @staticmethod
    def from_csv(csv_path, embedding_name: Optional[str] = None):
        """Create an instance of the SpeciesEmbedding class from a csv file.

        The first column of the csv file must contain the species and be named species.

        Args:
        ----
            csv_path (str): Filepath of the csv file
            embedding_name (str): The name of the species representation

        Returns:
        -------
            SpeciesEmbedding :class:`SpeciesEmbedding` instance.

        """
        # Need to add validation handling for csv files
        df = pd.read_csv(csv_path)
        species = list(df["species"])
        df = df.drop(["species"], axis=1)
        feature_labels = list(df.columns)
        embeds_array = df.to_numpy()
        embedding_data = {species[i]: embeds_array[i] for i in range(len(embeds_array))}
        return SpeciesEmbedding(embedding_data, embedding_name, feature_labels)

    @property
    def species_list(self) -> list:
        """Return the species of the embedding."""
        return list(self.embeddings.keys())

    @property
    def element_list(self) -> list:
        """Return the elements of the embedding."""
        return list({parse_species(species)[0] for species in self.species_list})

    def remove_species(self, species: Union[str, List[str]], inplace: bool = False):
        """Remove species from the SpeciesEmbedding instance.

        Args:
        ----
            species (str,list(str)): A species or a list of species
            inplace (bool): If True, species are removed
            from the SpeciesEmbedding instance.
            If false, the original SpeciesEmbedding instance is unchanged
            and a new SpeciesEmbedding instance with the species removed is created.

        """
        if inplace:
            if isinstance(species, str):
                del self.embeddings[species]
            elif isinstance(species, list):
                for sp in species:
                    del self.embeddings[sp]
            return None
        else:
            embeddings_copy = self.embeddings.copy()
            if isinstance(species, str):
                del embeddings_copy[species]
            elif isinstance(species, list):
                for sp in species:
                    del embeddings_copy[sp]
            return SpeciesEmbedding(embeddings_copy, self.embedding_name)

    def create_pairs(self):
        """Create all possible pairs of species."""
        return combinations_with_replacement(self.species_list, 2)

    @property
    def ion_type_dict(self) -> Dict[str, str]:
        """Return a dictionary of {species: ion type} pairs.

        e.g. {'Fe2+':'cation'}

        """
        ion_dict = {}
        for species in self.species_list:
            el, charge = parse_species(species)
            if charge > 0:
                ion_dict[species] = "cation"
            elif charge < 0:
                ion_dict[species] = "anion"
            else:
                ion_dict[species] = "neutral"

        return ion_dict

    @property
    def species_groups_dict(self) -> Dict[str, str]:
        """Return a dictionary of {species: element type} pairs.

        e.g. {'Fe2+':'transition metal'}

        """
        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[parse_species(i)[0]] for i in self.species_list}
