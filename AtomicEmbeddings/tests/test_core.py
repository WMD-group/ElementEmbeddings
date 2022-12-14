import unittest

import numpy as np
import pandas as pd

from AtomicEmbeddings import composition
from AtomicEmbeddings.core import Embedding


class TestSequenceFunctions(unittest.TestCase):
    # High Level functions

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

        pass
