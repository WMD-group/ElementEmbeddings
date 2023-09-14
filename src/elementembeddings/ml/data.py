"""Tools for creating and manipulating data sets."""

import abc
import os
from typing import Union

import torch
from torch.utils.data import Dataset

from elementembeddings.composition import composition_featuriser
from elementembeddings.core import Embedding
from elementembeddings.utils.config import DEFAULT_EMBEDDINGS

# from sklearn.preprocessing import StandardScaler,


class EmbeddingDataset(abc.ABC, Dataset):
    """Abstract class for material datasets.

    Wraps the data in a Dataset object.
    Provides methods for loading, processing and saving the data.
    Intended to automate featurisation of a dataset using embeddings.

    Executes the following steps:
    1. Load the data.
    2. Process the data.
    3. Save the data.
    4. Done.
    """

    def __init__(self, name: str, data=None, save_path=None):
        """Initialise the dataset.

        Args:
            name (str): Name of the dataset.
            data (list): List of data.
            save_path (str): Optional path to save the dataset to.
        """
        self.data = data
        self.name = name
        self.save_path = save_path
        self._load()

    @abc.abstractmethod
    def process(self):
        """Process the data."""
        pass

    def save(self):
        """Save the data."""
        pass

    def load(self):
        """Load the data."""
        pass

    def _load(self):
        """Load the data."""
        self.process()
        self.save()


class ElementEmbeddingDataset(EmbeddingDataset):
    """Dataset for material data which featurises data using element embeddings."""

    def __init__(
        self,
        formulas: list,
        targets: list,
        element_embeddings: Union[str, Embedding],
        stats: list = ["mean"],
        filename: str = None,
        name: str = "ElementEmbeddingDataset",
        transform=None,
    ):
        """Initialise the dataset.

        Args:
            formulas (list): List of formulas.
            target (list): List of target values.
            element_embeddings (str | Embedding): Element embeddings for features.
            Can either be a string of a default embedding,
            a path to a csv or json file or an Embedding object.
            stats (list): List of summary statistics to use for featurisation.
            filename (str): Optional filename to save the dataset to.
            transform (callable): Optional transform to be applied to the data.
        """
        self.formulas = formulas
        self.formula_vectors = None
        self.targets = targets
        self.mean = None
        self.std = None
        if os.path.exists(element_embeddings):
            if element_embeddings.endswith(".csv"):
                self.element_embeddings = Embedding.from_csv(element_embeddings)
            elif element_embeddings.endswith(".json"):
                self.element_embeddings = Embedding.from_json(element_embeddings)
        elif element_embeddings in DEFAULT_EMBEDDINGS.keys():
            self.element_embeddings = Embedding.load_data(element_embeddings)
        elif isinstance(element_embeddings, Embedding):
            self.element_embeddings = element_embeddings
        self.transform = transform
        self.stats = stats
        self.filename = filename

        super().__init__(name=name)

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        X = self.formula_vectors[idx]
        y = self.targets[idx]

        if self.transform:
            X = self.transform(X)

        return X, y

    def process(self):
        """Convert the formulas to feature vectors and the targets to tensors."""
        self.formula_vectors = composition_featuriser(
            data=self.formulas, embedding=self.element_embeddings, stats=self.stats
        )
        self.formula_vectors = torch.stack(
            [torch.from_numpy(vector) for vector in self.formula_vectors]
        ).to(torch.float32)
        self.targets = torch.tensor(self.targets).to(torch.float32)
        return self.formula_vectors
