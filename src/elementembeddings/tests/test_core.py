"""Test the core module of AtomicEmbeddings."""
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elementembeddings.core import Embedding


class EmbeddingTest(unittest.TestCase):
    """Test the Embedding class."""

    # High Level functions

    def test_Embedding_loading(self):
        """Test that the Embedding class can load the data."""
        skipatom = Embedding.load_data("skipatom")
        megnet16 = Embedding.load_data("megnet16")
        assert skipatom.dim == 200
        assert skipatom.embedding_name == "skipatom"
        assert megnet16.dim == 16
        assert megnet16.embedding_name == "megnet16"
        assert isinstance(skipatom.citation(), list)
        assert isinstance(megnet16.citation(), list)

    def test_Embeddings_class_magpie(self):
        """Test that the Embedding class can load the magpie data."""
        magpie = Embedding.load_data("magpie")
        # Check if the embeddings attribute is a dict
        assert isinstance(magpie.embeddings, dict)
        # Check if the embedding vector is a numpy array
        assert isinstance(magpie.embeddings["H"], np.ndarray)
        # Check if H is present in the embedding keys
        assert "H" in magpie.embeddings.keys()
        # Check dimensions
        assert magpie.dim == 22
        # Check that a list is returned
        assert isinstance(magpie.element_list, list)
        # Check that the correct list is returned
        el_list = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
        ]
        assert magpie.element_list == el_list
        # Check that a dictionary is returned
        assert isinstance(magpie.element_groups_dict, dict)
        # Check that the correct dictionary is returned
        group_dict = {
            "H": "Others",
            "He": "Noble gas",
            "Li": "Alkali",
            "Be": "Alkaline",
            "B": "Metalloid",
            "C": "Others",
            "N": "Others",
            "O": "Chalcogen",
            "F": "Halogen",
            "Ne": "Noble gas",
            "Na": "Alkali",
            "Mg": "Alkaline",
            "Al": "Post-TM",
            "Si": "Metalloid",
            "P": "Others",
            "S": "Chalcogen",
            "Cl": "Halogen",
            "Ar": "Noble gas",
            "K": "Alkali",
            "Ca": "Alkaline",
            "Sc": "TM",
            "Ti": "TM",
            "V": "TM",
            "Cr": "TM",
            "Mn": "TM",
            "Fe": "TM",
            "Co": "TM",
            "Ni": "TM",
            "Cu": "TM",
            "Zn": "TM",
            "Ga": "Post-TM",
            "Ge": "Metalloid",
            "As": "Metalloid",
            "Se": "Chalcogen",
            "Br": "Halogen",
            "Kr": "Noble gas",
            "Rb": "Alkali",
            "Sr": "Alkaline",
            "Y": "TM",
            "Zr": "TM",
            "Nb": "TM",
            "Mo": "TM",
            "Tc": "TM",
            "Ru": "TM",
            "Rh": "TM",
            "Pd": "TM",
            "Ag": "TM",
            "Cd": "TM",
            "In": "Post-TM",
            "Sn": "Post-TM",
            "Sb": "Metalloid",
            "Te": "Chalcogen",
            "I": "Halogen",
            "Xe": "Noble gas",
            "Cs": "Alkali",
            "Ba": "Alkaline",
            "La": "Lanthanoid",
            "Ce": "Lanthanoid",
            "Pr": "Lanthanoid",
            "Nd": "Lanthanoid",
            "Pm": "Lanthanoid",
            "Sm": "Lanthanoid",
            "Eu": "Lanthanoid",
            "Gd": "Lanthanoid",
            "Tb": "Lanthanoid",
            "Dy": "Lanthanoid",
            "Ho": "Lanthanoid",
            "Er": "Lanthanoid",
            "Tm": "Lanthanoid",
            "Yb": "Lanthanoid",
            "Lu": "Lanthanoid",
            "Hf": "TM",
            "Ta": "TM",
            "W": "TM",
            "Re": "TM",
            "Os": "TM",
            "Ir": "TM",
            "Pt": "TM",
            "Au": "TM",
            "Hg": "TM",
            "Tl": "Post-TM",
            "Pb": "Post-TM",
            "Bi": "Post-TM",
            "Po": "Chalcogen",
            "At": "Halogen",
            "Rn": "Noble gas",
            "Fr": "Alkali",
            "Ra": "Alkaline",
            "Ac": "Actinoid",
            "Th": "Actinoid",
            "Pa": "Actinoid",
            "U": "Actinoid",
            "Np": "Actinoid",
            "Pu": "Actinoid",
            "Am": "Actinoid",
            "Cm": "Actinoid",
            "Bk": "Actinoid",
        }
        assert magpie.element_groups_dict == group_dict
        # Check pair creation
        assert (
            len(list(magpie.create_pairs())) == 4753
        ), "Incorrect number of pairs returned"
        assert "H" not in magpie.remove_elements("H").element_list
        assert isinstance(magpie.citation(), list)
        assert isinstance(magpie.citation()[0], str)
        assert magpie._is_el_in_embedding("H")
        assert isinstance(magpie.correlation_df(), pd.DataFrame)

        # TO-DO
        # Create tests for checking dataframes and plotting functions
        assert isinstance(magpie.as_dataframe(), pd.DataFrame)
        assert isinstance(magpie.to(fmt="json"), str)
        assert isinstance(magpie.to(fmt="csv"), str)
        assert isinstance(
            magpie.compute_correlation_metric("H", "O", metric="pearson"),
            float,
        )
        assert isinstance(
            magpie.compute_distance_metric(
                "H",
                "O",
            ),
            float,
        )
        assert isinstance(magpie.distance_df(), pd.DataFrame)
        assert magpie.distance_df().shape == (
            len(list(magpie.create_pairs())) * 2 - len(magpie.embeddings),
            7,
        )
        assert magpie.distance_df().columns.tolist() == [
            "ele_1",
            "ele_2",
            "mend_1",
            "mend_2",
            "Z_1",
            "Z_2",
            "euclidean",
        ]
        assert isinstance(magpie.distance_pivot_table(), pd.DataFrame)
        assert isinstance(magpie.plot_distance_correlation(), plt.Axes)
        assert isinstance(
            magpie.plot_distance_correlation(metric="euclidean"), plt.Axes
        )
