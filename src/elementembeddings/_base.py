import json
import random
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP


class EmbeddingBase(ABC):
    """Base class for embeddings."""

    def __init__(
        self,
        embeddings: dict,
        embedding_name: Optional[str] = None,
        feature_labels: Optional[List[str]] = None,
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
            self.embedding_type: str = "one-hot"

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
    def from_json(json_path: str, embedding_name: Optional[str] = None):
        """Create an embedding from a json file.

        Args:
        ----
            json_path (str): The path to the json file.
            embedding_name (str): The name of the embedding.
        """
        with open(json_path) as f:
            embeddings = json.load(f)
        return EmbeddingBase(embeddings, embedding_name)

    def _is_standardised(self):
        """Check if the embedding is standardised.

        Mean must be 0 and standard deviation must be 1.
        """
        return np.isclose(
            np.mean(np.array(list(self.embeddings.values()))),
            0,
        ) and np.isclose(np.std(np.array(list(self.embeddings.values()))), 1)

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
                "Embedding is already standardised."
                "Returning None and not changing the embedding",
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
                self = self.standardise()
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
                self = self.standardise()
                embeddings_array = np.array(list(self.embeddings.values()))
        else:
            warnings.warn(
                """It is recommended to scale the embeddings
                before projecting with t-SNE.
                To do so, set `standardise=True`.""",
            )
            embeddings_array = np.array(list(self.embeddings.values()))

        tsne = TSNE(n_components=n_components, **kwargs)
        return tsne.fit_transform(embeddings_array)

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
                self = self.standardise()
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
