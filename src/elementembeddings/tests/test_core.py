"""Test the core module of AtomicEmbeddings."""
import copy
import os
import unittest

import numpy as np
import pandas as pd

from elementembeddings.core import Embedding

test_files_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "files")
TEST_EMBEDDING_CSV = os.path.join(test_files_dir, "test_embedding.csv")
TEST_EMBEDDING_JSON = os.path.join(test_files_dir, "test_embedding.json")


class EmbeddingTest(unittest.TestCase):
    """Test the Embedding class."""

    # High Level functions
    @classmethod
    def setUpClass(cls):
        """Set up the test."""
        cls.test_skipatom = Embedding.load_data("skipatom")
        cls.test_megnet16 = Embedding.load_data("megnet16")
        cls.test_matscholar = Embedding.load_data("matscholar")
        cls.test_mod_petti = Embedding.load_data("mod_petti")
        cls.test_magpie = Embedding.load_data("magpie")
        cls.test_atomic = Embedding.load_data("atomic")
        cls.test_magpie_sc = Embedding.load_data("magpie_sc")

    def test_Embedding_attributes(self):
        """Test attributes of the loaded embeddings."""
        assert self.test_skipatom.dim == 200
        assert self.test_skipatom.embedding_name == "skipatom"
        assert self.test_skipatom.embedding_type == "vector"
        assert self.test_megnet16.dim == 16
        assert self.test_megnet16.embedding_name == "megnet16"
        assert self.test_megnet16.embedding_type == "vector"
        assert self.test_matscholar.dim == 200
        assert self.test_matscholar.embedding_name == "matscholar"
        assert self.test_matscholar.embedding_type == "vector"
        assert self.test_mod_petti.dim == 103
        assert self.test_mod_petti.embedding_name == "mod_petti"
        assert self.test_mod_petti.embedding_type == "linear"
        assert isinstance(self.test_skipatom.citation(), list)
        assert isinstance(self.test_megnet16.citation(), list)
        assert isinstance(self.test_matscholar.citation(), list)
        assert isinstance(self.test_mod_petti.citation(), list)

    def test_Embedding_standardised(self):
        """Test that the Embedding class can check if the data is standardised."""
        assert self.test_magpie.is_standardised is False
        assert self.test_magpie_sc.is_standardised is True

    def test_Embedding_standardisation(self):
        """Test the standardisation method of the Embedding class."""
        assert self.test_magpie.is_standardised is False
        assert self.test_magpie.standardise().is_standardised is True
        assert self.test_skipatom.is_standardised is False
        assert self.test_skipatom.standardise().is_standardised is True
        assert self.test_skipatom.standardise().standardise() is None
        assert copy.deepcopy(self.test_magpie).standardise(inplace=True) is None

    def test_Embedding_file_input(self):
        """Test that the Embedding class can load custom data."""
        embedding_csv = Embedding.from_csv(TEST_EMBEDDING_CSV)
        embedding_json = Embedding.from_json(TEST_EMBEDDING_JSON)
        assert embedding_csv.dim == 10
        assert embedding_json.dim == 10

    def test_Embedding_class_mod_petti(self):
        """Test that the Embedding class can load the mod_petti data."""
        mod_petti = self.test_mod_petti
        # Check if the embeddings attribute is a dict
        assert isinstance(mod_petti.embeddings, dict)
        # Check if the embedding vector is a numpy array
        assert isinstance(mod_petti.embeddings["H"], np.ndarray)
        # Check if H is present in the embedding keys
        assert "H" in mod_petti.embeddings.keys()
        # Check dimensions
        assert mod_petti.dim == 103
        # Check embedding type
        assert mod_petti.embedding_type == "linear"
        # Check that a list is returned
        assert isinstance(mod_petti.element_list, list)
        # Check the the dimensons of the embedding vector
        assert mod_petti.embeddings["H"].shape == (103,)
        # Check that the embedding vector is not all zeros
        assert not np.all(mod_petti.embeddings["H"] == 0)
        # Check the the embedding vector for H is correct
        test_H = np.zeros(103)
        test_H[-1] = 1
        assert np.all(mod_petti.embeddings["H"] == test_H)

    def test_Embedding_class_atomic(self):
        """Test that the Embedding class can load the atomic data."""
        atomic = self.test_atomic
        # Check if the embeddings attribute is a dict
        assert isinstance(atomic.embeddings, dict)
        # Check if the embedding vector is a numpy array
        assert isinstance(atomic.embeddings["H"], np.ndarray)
        # Check if H is present in the embedding keys
        assert "H" in atomic.embeddings.keys()
        # Check dimensions
        assert atomic.dim == 118
        # Check embedding type
        assert atomic.embedding_type == "linear"
        # Check that a list is returned
        assert isinstance(atomic.element_list, list)
        # Check the the dimensons of the embedding vector
        assert atomic.embeddings["H"].shape == (118,)
        # Check that the embedding vector is not all zeros
        assert not np.all(atomic.embeddings["H"] == 0)
        # Check the the embedding vector for H is correct
        test_H = np.zeros(118)
        test_H[0] = 1
        assert np.all(atomic.embeddings["H"] == test_H)

    def test_Embedding_class_magpie(self):
        """Test that the Embedding class can load the magpie data."""
        magpie = self.test_magpie
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

    def test_as_dataframe(self):
        """Test the as_dataframe method."""
        magpie = self.test_magpie
        assert isinstance(magpie.as_dataframe(), pd.DataFrame)
        assert "H" in magpie.as_dataframe().index.tolist()
        assert isinstance(magpie.as_dataframe(columns="elements"), pd.DataFrame)
        assert "H" in magpie.as_dataframe(columns="elements").columns.tolist()
        self.assertRaises(ValueError, magpie.as_dataframe, columns="test")

    def test_to(self):
        """Test the to method."""
        assert isinstance(self.test_magpie.to(fmt="json"), str)
        self.test_magpie.to(fmt="json", filename="test.json")
        assert os.path.isfile("test.json")
        os.remove("test.json")

        assert isinstance(self.test_magpie.to(fmt="csv"), str)
        self.test_magpie.to(fmt="csv", filename="test.csv")
        assert os.path.isfile("test.csv")
        os.remove("test.csv")

    def test_compute_metric_functions(self):
        """Test the compute metric functions."""
        assert isinstance(
            self.test_magpie.compute_correlation_metric("H", "O", metric="pearson"),
            float,
        )
        assert isinstance(
            self.test_magpie.compute_distance_metric(
                "H",
                "O",
            ),
            float,
        )
        assert isinstance(
            self.test_magpie.compute_distance_metric("H", "O", "energy"),
            float,
        )
        assert isinstance(
            self.test_magpie.compute_distance_metric("H", "O", "cosine_distance"),
            float,
        )
        assert isinstance(
            self.test_magpie.compute_correlation_metric("H", "O", metric="spearman"),
            float,
        )

        self.assertRaises(
            ValueError, self.test_skipatom.compute_distance_metric, "He", "O"
        )
        self.assertRaises(
            ValueError, self.test_skipatom.compute_distance_metric, "O", "He"
        )
        self.assertRaises(
            ValueError, self.test_skipatom.compute_distance_metric, "Li", "O", "euclid"
        )

    def test_distance_dataframe_functions(self):
        """Test the distance dataframe functions."""
        assert isinstance(self.test_magpie.distance_df(), pd.DataFrame)
        assert self.test_magpie.distance_df().shape == (
            len(list(self.test_magpie.create_pairs())) * 2
            - len(self.test_magpie.embeddings),
            7,
        )
        assert self.test_magpie.distance_df().columns.tolist() == [
            "ele_1",
            "ele_2",
            "mend_1",
            "mend_2",
            "Z_1",
            "Z_2",
            "euclidean",
        ]
        assert isinstance(self.test_magpie.distance_pivot_table(), pd.DataFrame)
        assert isinstance(
            self.test_magpie.distance_pivot_table(sortby="atomic_number"), pd.DataFrame
        )

    def test_remove_elements(self):
        """Test the remove_elements function."""
        assert isinstance(self.test_skipatom.remove_elements("H"), Embedding)
        assert isinstance(self.test_skipatom.remove_elements(["H", "Li"]), Embedding)
        self.assertIsNone(self.test_skipatom.remove_elements("H", inplace=True))
        self.assertFalse(self.test_skipatom._is_el_in_embedding("H"))
        self.assertIsNone(
            self.test_skipatom.remove_elements(["Li", "Ti", "Bi"], inplace=True)
        )
        assert "Li" not in self.test_skipatom.element_list
        assert "Ti" not in self.test_skipatom.element_list
        assert "Bi" not in self.test_skipatom.element_list

    def test_PCA(self):
        """Test the PCA function."""
        assert isinstance(self.test_matscholar.calculate_PC(), np.ndarray)
        assert self.test_matscholar.calculate_PC().shape == (
            len(self.test_matscholar.element_list),
            2,
        )

    def test_tSNE(self):
        """Test the tSNE function."""
        assert isinstance(self.test_matscholar.calculate_tSNE(), np.ndarray)
        assert self.test_matscholar.calculate_tSNE().shape == (
            len(self.test_matscholar.element_list),
            2,
        )

    def test_UMAP(self):
        """Test the UMAP function."""
        assert isinstance(self.test_matscholar.calculate_UMAP(), np.ndarray)
        assert self.test_matscholar.calculate_UMAP().shape == (
            len(self.test_matscholar.element_list),
            2,
        )
