"""Provides the `Embedding` class.

This module enables the user load in elemental representation data
and analyse it using statistical functions.

Typical usage example:
    megnet16 = Embedding.load_data('megnet16')
"""

from __future__ import annotations

import fnmatch
import json
import warnings
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ._base import EmbeddingBase
from .utils.config import DEFAULT_ELEMENT_EMBEDDINGS, DEFAULT_SPECIES_EMBEDDINGS
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

    @staticmethod
    def load_data(embedding_name: str | None = None):
        """Create an instance of the `Embedding` class from a default embedding file.

        The default embeddings are in the table below:

        | **Name**                | **str_name** |
        |-------------------------|--------------|
        | Magpie                  | magpie       |
        | Magpie (scaled)         | magpie_sc    |
        | Mat2Vec                 | mat2vec      |
        | Matscholar              | matscholar   |
        | Megnet (16 dimensions)  | megnet16     |
        | Modified Pettifor scale | mod_petti    |
        | Oliynyk                 | oliynyk      |
        | Oliynyk (scaled)        | oliynyk_sc   |
        | Random (200 dimensions) | random_200   |
        | SkipAtom                | skipatom     |
        | Atomic Number           | atomic       |
        | CrystaLLM               | crystallm    |
        | XenonPy                 | xenonpy      |
        | Cgnf                    | cgnf         |


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
    def from_json(embedding_json, embedding_name: str | None = None):
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
    def from_csv(embedding_csv, embedding_name: str | None = None):
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
        embedding_data = {elements[i]: embeds_array[i] for i in range(len(embeds_array))}
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
            msg = f"{columns} is not a valid keyword argument. " f"Choose either 'components' or 'elements"
            raise (
                ValueError(
                    msg,
                )
            )

    def to(self, fmt: str = "", filename: str | None = ""):
        """Output the embedding to a file.

        Args:
        ----
            fmt (str): The file format to output the embedding to.
                Options include "json" and "csv".
            filename (str): The name of the file to be outputted

        Returns:
        -------
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

    def remove_elements(self, elements: str | list[str], inplace: bool = False):
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
                "Embedding is already standardised. " "Returning None and not changing the embedding.",
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

    @property
    def element_groups_dict(self) -> dict[str, str]:
        """Return a dictionary of {element: element type} pairs.

        e.g. {'He':'Noble gas'}

        """
        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[i] for i in self.element_list}


class SpeciesEmbedding(EmbeddingBase):
    """Represent an ion representation.

    To load an embedding distributed from the package use the load_data() method.

    Works like a standard python dictionary. The keys are {species: vector} pairs.
    """

    @staticmethod
    def load_data(embedding_name: str, include_neutral: bool = False):
        """Create a `SpeciesEmbedding` from a preset embedding file.

        The default embeddings are in the table below:

        | **Name**                | **str_name** |
        |-------------------------|--------------|
        | SkipSpecies (200 dim, MPv2022)             | skipspecies  |
        | SkipSpecies (induced, 200 dim, MPv2022)   | skipspecies_induced |

        Args:
        ----
            embedding_name (str): The str_name of the species representation
            include_neutral (bool): If True, neutral species are
                included in the embedding

        Returns:
        -------
            SpeciesEmbedding :class:`SpeciesEmbedding` instance.
        """
        if DEFAULT_SPECIES_EMBEDDINGS[embedding_name].endswith(".csv"):
            embedding = SpeciesEmbedding.from_csv(
                path.join(
                    data_directory,
                    "species_representations",
                    DEFAULT_SPECIES_EMBEDDINGS[embedding_name],
                ),
                embedding_name,
            )
            if not include_neutral:
                embedding.remove_neutral_species(inplace=True)
            return embedding
        elif DEFAULT_SPECIES_EMBEDDINGS[embedding_name].endswith(".json"):
            embedding = SpeciesEmbedding.from_json(
                path.join(
                    data_directory,
                    "species_representations",
                    DEFAULT_SPECIES_EMBEDDINGS[embedding_name],
                ),
                embedding_name,
            )
            if not include_neutral:
                embedding.remove_neutral_species(inplace=True)
            return embedding
        else:
            return None

    @staticmethod
    def from_csv(csv_path, embedding_name: str | None = None):
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

    @staticmethod
    def from_json(json_path, embedding_name: str | None = None):
        """Create an instance of the SpeciesEmbedding class from a json file.

        Args:
        ----
            json_path (str): Filepath of the json file
            embedding_name (str): The name of the species representation

        Returns:
        -------
            SpeciesEmbedding :class:`SpeciesEmbedding` instance.

        """
        # Need to add validation handling for json files
        with open(json_path) as f:
            embedding_data = json.load(f)
        return SpeciesEmbedding(embedding_data, embedding_name)

    @property
    def species_list(self) -> list:
        """Return the species of the embedding."""
        return list(self.embeddings.keys())

    @property
    def element_list(self) -> list:
        """Return the elements of the embedding."""
        return list({parse_species(species)[0] for species in self.species_list})

    def remove_neutral_species(self, inplace: bool = False):
        """Remove neutral species from the SpeciesEmbedding instance.

        Args:
        ----
            inplace (bool): If True, neutral species are removed
                from the SpeciesEmbedding instance.
            If false, the original SpeciesEmbedding instance is unchanged
            and a new SpeciesEmbedding instance with the
                neutral species removed is created.

        """
        neutral_species = [s for s in self.species_list if parse_species(s)[1] == 0]
        return self.remove_species(neutral_species, inplace)

    def get_element_oxi_states(self, el: str) -> list:
        """Return the oxidation states for a given element.

        Args:
        ----
            el (str): An element symbol

        Returns:
        -------
            oxidation_states (list[int]): A list of oxidation states
        """
        assert el in self.element_list, f"There are no species of the element {el} in this SpeciesEmbedding"
        parsed_species = [parse_species(species) for species in self.species_list]

        el_species_list = [species for species in parsed_species if species[0] == el]
        oxidation_states = [species[1] for species in el_species_list]
        return sorted(oxidation_states)

    def remove_species(self, species: str | list[str], inplace: bool = False):
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
                try:
                    del self.embeddings[species]
                except KeyError:
                    warnings.warn(
                        f"{species} is not in the SpeciesEmbedding. " "Skipping this species.",
                    )
            elif isinstance(species, list):
                for sp in species:
                    try:
                        del self.embeddings[sp]
                    except KeyError:
                        warnings.warn(
                            f"{sp} is not in the SpeciesEmbedding. " "Skipping this species.",
                        )
            return None
        else:
            embeddings_copy = self.embeddings.copy()
            if isinstance(species, str):
                try:
                    del embeddings_copy[species]
                except KeyError:
                    warnings.warn(
                        f"{species} is not in the SpeciesEmbedding. " "Skipping this species.",
                    )
            elif isinstance(species, list):
                for sp in species:
                    try:
                        del embeddings_copy[sp]
                    except KeyError:
                        warnings.warn(
                            f"{sp} is not in the SpeciesEmbedding. " "Skipping this species.",
                        )
            return SpeciesEmbedding(embeddings_copy, self.embedding_name)

    @property
    def ion_type_dict(self) -> dict[str, str]:
        """Return a dictionary of {species: ion type} pairs.

        e.g. {'Fe2+':'cation'}

        """
        ion_dict = {}
        for species in self.species_list:
            el, charge = parse_species(species)
            if charge > 0:
                ion_dict[species] = "Cation"
            elif charge < 0:
                ion_dict[species] = "Anion"
            else:
                ion_dict[species] = "Neutral"

        return ion_dict

    @property
    def species_groups_dict(self) -> dict[str, str]:
        """Return a dictionary of {species: element type} pairs.

        e.g. {'Fe2+':'transition metal'}

        """
        with open(path.join(data_directory, "element_data/element_group.json")) as f:
            _dict = json.load(f)
        return {i: _dict[parse_species(i)[0]] for i in self.species_list}

    def distance_df(self, metric="euclidean") -> pd.DataFrame:
        """Return a dataframe of the distance between species.

        Args:
        ----
            metric (str): The metric to use to calculate the distance.
            Options are 'euclidean', 'cosine', 'manhattan' and 'chebyshev'.

        Returns:
        -------
            df (pandas.DataFrame): A pandas dataframe object
        """
        return super().distance_df(metric).rename(mapper={"ele_1": "species_1", "ele_2": "species_2"}, axis=1)

    def correlation_df(self, metric: str = "pearson") -> pd.DataFrame:
        """Return a dataframe of the correlation between species.

        Args:
        ----
            metric (str): The metric to use to calculate the correlation.
            Options are 'pearson' and 'spearman'.

        Returns:
        -------
            df (pandas.DataFrame): A pandas dataframe object

        """
        return super().correlation_df(metric).rename(mapper={"ele_1": "species_1", "ele_2": "species_2"}, axis=1)

    def to(self, fmt: str = "", filename: str | None = ""):
        """Output the embedding to a file.

        Args:
        ----
            fmt (str): The file format to output the embedding to.
                Options include "json" and "csv".
            filename (str): The name of the file to be outputted

        Returns:
        -------
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
        return None
