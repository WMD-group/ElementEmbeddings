import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats._result_classes import PearsonRResult

from AtomicEmbeddings import composition
from AtomicEmbeddings.core import Embedding


class TestSequenceFunctions(unittest.TestCase):
    # High Level functions

    def test_Embedding_loading(self):
        # Test that the skipatom and megnet16 can be loaded
        skipatom = composition.Embedding.load_data("skipatom")
        megnet16 = composition.Embedding.load_data("megnet16")
        self.assertEqual(skipatom.dim, 200)
        self.assertEqual(skipatom.embedding_name, "skipatom")
        self.assertEqual(megnet16.dim, 16)
        self.assertEqual(megnet16.embedding_name, "megnet16")
        self.assertIsInstance(skipatom.citation(), list)
        self.assertIsInstance(megnet16.citation(), list)

    def test_Embeddings_class_magpie(self):
        magpie = Embedding.load_data("magpie")
        # Check if the embeddings attribute is a dict
        self.assertIsInstance(magpie.embeddings, dict)
        # Check if the embedding vector is a numpy array
        self.assertIsInstance(magpie.embeddings["H"], np.ndarray)
        # Check if H is present in the embedding keys
        self.assertIn("H", magpie.embeddings.keys())
        # Check dimensions
        self.assertEqual(magpie.dim, 21)
        # Check that a list is returned
        self.assertIsInstance(magpie.element_list, list)
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
        self.assertEqual(magpie.element_list, el_list)
        # Check that a dictionary is returned
        self.assertIsInstance(magpie.element_groups_dict, dict)
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
        self.assertEqual(magpie.element_groups_dict, group_dict)
        # Check pair creation
        self.assertEqual(
            len(list(magpie.create_pairs())), 4753, "Incorrect number of pairs returned"
        )
        self.assertTrue("H" not in magpie.remove_elements("H").element_list)
        self.assertIsInstance(magpie.citation(), list)
        self.assertIsInstance(magpie.citation()[0], str)
        self.assertTrue(magpie._is_el_in_embedding("H"))
        self.assertIsInstance(magpie.create_correlation_df(), pd.DataFrame)

        # TO-DO
        # Create tests for checking dataframes and plotting functions
        self.assertIsInstance(magpie.as_dataframe(), pd.DataFrame)
        self.assertIsInstance(magpie.to(fmt="json"), str)
        self.assertIsInstance(magpie.to(fmt="csv"), str)
        self.assertIsInstance(
            magpie.compute_correlation_metric("H", "O", metric="pearson"),
            PearsonRResult,
        )
        self.assertIsInstance(
            magpie.compute_distance_metric(
                "H",
                "O",
            ),
            float,
        )
        self.assertIsInstance(magpie.create_distance_correlation_df(), pd.DataFrame)
        self.assertEqual(
            magpie.create_distance_correlation_df().shape,
            (len(list(magpie.create_pairs())) * 2 - len(magpie.embeddings), 5),
        )
        self.assertListEqual(
            magpie.create_distance_correlation_df().columns.tolist(),
            ["ele_1", "ele_2", "mend_1", "mend_2", "euclidean"],
        )
        self.assertIsInstance(magpie.create_distance_pivot_table(), pd.DataFrame)
        self.assertIsInstance(magpie.plot_distance_correlation(), plt.Axes)
        self.assertIsInstance(
            magpie.plot_distance_correlation(metric="euclidean"), plt.Axes
        )

    # ------------ Compositon.py functions ------------
    def test_formula_parser(self):
        LLZO_parsed = composition.formula_parser("Li7La3ZrO12")
        self.assertIsInstance(LLZO_parsed, dict)
        self.assertTrue("Zr" in LLZO_parsed)
        self.assertEqual(LLZO_parsed["Li"], 7)

    def test__get_fractional_composition(self):
        CsPbI3_frac = composition._get_fractional_composition("CsPbI3")
        self.assertIsInstance(CsPbI3_frac, dict)
        self.assertTrue("Pb" in CsPbI3_frac)
        self.assertEqual(CsPbI3_frac["I"], 0.6)

    def test_Composition_class(self):
        Fe2O3_magpie = composition.CompositionalEmbedding(
            formula="Fe2O3", embedding="magpie"
        )
        self.assertIsInstance(Fe2O3_magpie.embedding, Embedding)
        self.assertEqual(Fe2O3_magpie.formula, "Fe2O3")
        self.assertEqual(Fe2O3_magpie.embedding_name, "magpie")
        self.assertIsInstance(Fe2O3_magpie.composition, dict)
        self.assertAlmostEqual({"Fe": 2, "O": 3}, Fe2O3_magpie.composition)
        self.assertEqual(Fe2O3_magpie._natoms, 5)
        self.assertEqual(Fe2O3_magpie.fractional_composition, {"Fe": 0.4, "O": 0.6})
        self.assertIsInstance(Fe2O3_magpie._mean_feature_vector(), np.ndarray)
        # Test that the feature vector function works
        stats = [
            "mean",
            "variance",
            "minpool",
            "maxpool",
            "sum",
            "range",
            "geometric_mean",
            "harmonic_mean",
        ]
        self.assertIsInstance(Fe2O3_magpie.feature_vector(stats=stats), np.ndarray)
        self.assertEqual(
            len(Fe2O3_magpie.feature_vector(stats=stats)),
            Fe2O3_magpie.embedding.dim * len(stats),
        )
        # Test that the feature vector function works with a single stat
        self.assertIsInstance(Fe2O3_magpie.feature_vector(stats="mean"), np.ndarray)

    def test_composition_featuriser(self):
        formulas = ["Fe2O3", "Li7La3ZrO12", "CsPbI3"]
        formula_df = pd.DataFrame(formulas, columns=["formula"])
        self.assertIsInstance(
            composition.composition_featuriser(formula_df), pd.DataFrame
        )
        self.assertEqual(composition.composition_featuriser(formula_df).shape, (3, 2))
        self.assertIsInstance(composition.composition_featuriser(formulas), list)
        self.assertEqual(len(composition.composition_featuriser(formulas)), 3)
