"""Test the classes and functions in the composition module."""
import unittest

import numpy as np
import pandas as pd

from elementembeddings import composition, core


# ------------ Compositon.py functions ------------
class TestComposition(unittest.TestCase):
    """Test the composition module."""

    def test_formula_parser(self):
        """Test the formula_parser function."""
        LLZO_parsed = composition.formula_parser("Li7La3ZrO12")
        assert isinstance(LLZO_parsed, dict)
        assert "Zr" in LLZO_parsed
        assert LLZO_parsed["Li"] == 7

    def test__get_fractional_composition(self):
        """Test the _get_fractional_composition function."""
        CsPbI3_frac = composition._get_fractional_composition("CsPbI3")
        assert isinstance(CsPbI3_frac, dict)
        assert "Pb" in CsPbI3_frac
        assert CsPbI3_frac["I"] == 0.6

    def test_Composition_class(self):
        """Test the Composition class."""
        Fe2O3_magpie = composition.CompositionalEmbedding(
            formula="Fe2O3", embedding="magpie"
        )
        assert isinstance(Fe2O3_magpie.embedding, core.Embedding)
        assert Fe2O3_magpie.formula == "Fe2O3"
        assert Fe2O3_magpie.embedding_name == "magpie"
        assert isinstance(Fe2O3_magpie.composition, dict)
        assert {"Fe": 2, "O": 3} == Fe2O3_magpie.composition
        assert Fe2O3_magpie._natoms == 5
        assert Fe2O3_magpie.fractional_composition == {"Fe": 0.4, "O": 0.6}
        assert isinstance(Fe2O3_magpie._mean_feature_vector(), np.ndarray)
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
        assert isinstance(Fe2O3_magpie.feature_vector(stats=stats), np.ndarray)
        assert len(
            Fe2O3_magpie.feature_vector(stats=stats)
        ) == Fe2O3_magpie.embedding.dim * len(stats)
        # Test that the feature vector function works with a single stat
        assert isinstance(Fe2O3_magpie.feature_vector(stats="mean"), np.ndarray)

    def test_composition_featuriser(self):
        """Test the composition featuriser function."""
        formulas = ["Fe2O3", "Li7La3ZrO12", "CsPbI3"]
        formula_df = pd.DataFrame(formulas, columns=["formula"])
        assert isinstance(composition.composition_featuriser(formula_df), pd.DataFrame)
        assert composition.composition_featuriser(formula_df).shape == (3, 2)
        assert isinstance(composition.composition_featuriser(formulas), list)
        assert len(composition.composition_featuriser(formulas)) == 3
