# Contains functions for making CBFVs
import collections
import re
from typing import Dict, Generator, Iterator, Union, cast

import numpy as np

from .core import Embedding

# Modified from pymatgen.core.Compositions


def formula_parser(formula: str) -> Dict[str, float]:
    # TO-DO: Need to add validation to check composition contains real elements
    """
    Parses a string formula and returns a dictionary of the composition with key:value pairs of element symbol: amount

    Args:
        formula (str): A string formula e.g. CsPbI3, Li7La3Zr2O12

    Returns:
        (dict): A dictionary of the composition

    """

    # For Metallofullerene
    formula = formula.replace("@", "")

    regex = r"\(([^\(\)]+)\)\s*([\.e\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1.0
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return formula_parser(expanded_formula)
    return get_sym_dict(formula, 1)


# Parses formula and returns a dictionary of ele symbol: amount
# From pymatgen.core.composition
def get_sym_dict(formula: str, factor: Union[int, float]) -> Dict[str, float]:
    sym_dict: Dict[str, float] = collections.defaultdict(float)
    regex = r"([A-Z][a-z]*)\s*([-*\.e\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, formula):
        el = m.group(1)
        amt = 1.0
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        formula = formula.replace(m.group(), "", 1)
    if formula.strip():
        raise ValueError(f"{formula} is an invalid formula")

    return sym_dict


# Function for fractional compositions
def _get_fractional_composition(formula: str) -> Dict[str, float]:
    el_dict = formula_parser(formula)
    elamt = {}
    natoms = 0
    for el, v in el_dict.items():
        elamt[el] = v
        natoms += abs(v)
    return {el: elamt[el] / natoms for el in elamt}


# Class to handle compositional embeddings
class CompositionalEmbedding:
    def __init__(self, formula: str, embedding: Union[str, Embedding], x=1):
        self.embedding = embedding

        # If a string has been passed for embedding, create an Embedding instance
        if isinstance(embedding, str):
            self.embedding = Embedding.load_data(embedding)

        self.embedding_name: str = self.embedding.embedding_name
        # Set an attribute for the formula
        self.formula = formula

        # Set an attribute for the comp dict
        comp_dict = formula_parser(self.formula)
        self._natoms = 0
        for el, v in comp_dict.items():
            if v < 0:
                raise ValueError("Formula cannot contain negative amounts of elements")
            self._natoms += abs(v)

        self.composition = comp_dict

    @property
    def fractional_composition(self):
        return _get_fractional_composition(self.formula)

    @property
    def num_atoms(self) -> float:
        """
        Total number of atoms in Composition
        """
        return self._natoms

    def as_dict(self) -> dict:
        # TO-DO: Need to create a dict representation for the embedding class
        """
        Returns the CompositionalEmbedding class as a dict
        """
        return {
            "formula": self.formula,
            "composition": self.composition,
            "fractional_composition": self.fractional_composition,
            # 'embedding':self.embedding.as_
        }
        # Se

        # Set an attribute
        pass

    def _mean_feature_vector(self) -> np.ndarray:
        """
        Computes a weighted mean feature vector based of the embedding. The dimension of the feature vector is the same as the embedding.

        """
        n = int(len(self.fractional_composition))
        m = len(self.embedding.embeddings["H"])
        el_matrix = np.zeros(shape=(n, m))
        for i, k in enumerate(self.fractional_composition.keys()):
            el_matrix[i] = self.embedding.embeddings[k]

        el_matrix = np.nan_to_num(el_matrix)

        return np.dot(np.array(list(self.fractional_composition.values())), el_matrix)

    def _variance_feature_vector(self) -> np.ndarray:
        """
        Computes a weighted variance feature vector
        """
        pass

    def _minpool_feature_vector(self) -> np.ndarray:
        """
        Computes a min pooled feature vector
        """
        pass

    def _maxpool_feature_vector(self) -> np.ndarray:
        """
        Computes a max pooled feature vector
        """
        pass

    def _range_feature_vector(self) -> np.ndarray:
        """
        Computes a range feature vector
        """
        pass

    def _sum_feature_vector(self) -> np.ndarray:
        """
        Computes the weighted sum feature vector
        """

        n = int(len(self.composition))
        m = len(self.embedding.embeddings["H"])
        el_matrix = np.zeros(shape=(n, m))
        for i, k in enumerate(self.composition.keys()):
            el_matrix[i] = self.embedding.embeddings[k]

        el_matrix = np.nan_to_num(el_matrix)

        return np.dot(np.array(list(self.composition.values())), el_matrix)

    def _geometric_mean_feature_vector(self) -> np.ndarray:
        pass

    def _harmonic_mean_feature_vector(self) -> np.ndarray:
        pass

    pass
