"""Test the classes and functions in the composition module."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pytest

from elementembeddings import composition, core


# ------------ Compositon.py functions ------------
class TestComposition(unittest.TestCase):
    """Test the composition module."""

    def setUp(self):
        """Set up the test class."""
        self.formulas = [
            "Sr3Sc2(GeO4)3",
            "Fe2O3",
            "Li7La3ZrO12",
            "CsPbI3",
        ]

    def test_formula_parser(self):
        """Test the formula_parser function."""
        LLZO_parsed = composition.formula_parser(self.formulas[2])
        assert isinstance(LLZO_parsed, dict)
        assert "Zr" in LLZO_parsed
        assert LLZO_parsed["Li"] == 7

    def test_formula_parser_with_parentheses(self):
        """Test the formula_parser function with parentheses."""
        SrScGeO4_parsed = composition.formula_parser(self.formulas[0])
        assert isinstance(SrScGeO4_parsed, dict)
        assert "Sr" in SrScGeO4_parsed
        assert SrScGeO4_parsed["Ge"] == 3

    def test_formula_parser_with_invalid_formula(self):
        """Test the formula_parser function with an invalid formula."""
        with pytest.raises(ValueError):
            composition.formula_parser("Sr3Sc2(GeO4)3)")

    def test__get_fractional_composition(self):
        """Test the _get_fractional_composition function."""
        CsPbI3_frac = composition._get_fractional_composition(self.formulas[3])
        assert isinstance(CsPbI3_frac, dict)
        assert "Pb" in CsPbI3_frac
        assert CsPbI3_frac["I"] == 0.6


class TestCompositionalEmbedding(unittest.TestCase):
    """Test the CompositionalEmbedding class."""

    def setUp(self):
        """Set up the test formulas."""
        self.formulas = ["Sr3Sc2(GeO4)3", "Fe2O3", "Li7La3ZrO12", "CsPbI3", "CsPbI-3"]
        self.valid_magpie_compositions = [
            composition.CompositionalEmbedding(formula=formula, embedding="magpie") for formula in self.formulas[:3]
        ]
        self.stats = [
            "mean",
            "variance",
            "minpool",
            "maxpool",
            "sum",
            "range",
            "geometric_mean",
            "harmonic_mean",
        ]

    def test_CompositionalEmbedding_attributes(self):
        """Test the Composition class."""
        Fe2O3_magpie = self.valid_magpie_compositions[1]
        assert isinstance(Fe2O3_magpie.embedding, core.Embedding)
        assert Fe2O3_magpie.formula == "Fe2O3"
        assert Fe2O3_magpie.embedding_name == "magpie"
        assert isinstance(Fe2O3_magpie.composition, dict)
        assert Fe2O3_magpie.composition == {"Fe": 2, "O": 3}
        assert Fe2O3_magpie.num_atoms == 5
        assert Fe2O3_magpie.fractional_composition == {"Fe": 0.4, "O": 0.6}
        assert Fe2O3_magpie.embedding.dim == 22

    def test_CompositionalEmbedding_negative_formula(self):
        """Test the Composition class with a negative formula."""
        with pytest.raises(ValueError):
            composition.CompositionalEmbedding(
                formula=self.formulas[4],
                embedding="magpie",
            )

    def test_CompositionalEmbedding_as_dict(self):
        """Test the Composition class as a dictionary."""
        assert isinstance(self.valid_magpie_compositions[0].as_dict(), dict)
        assert self.valid_magpie_compositions[0].as_dict() == {
            "formula": "Sr3Sc2(GeO4)3",
            "composition": {"Sr": 3, "Sc": 2, "Ge": 3, "O": 12},
            "fractional_composition": {"Sr": 0.15, "Sc": 0.1, "Ge": 0.15, "O": 0.6},
        }

    def test__mean_feature_vector(self):
        """Test the _mean_feature_vector function."""
        assert isinstance(
            self.valid_magpie_compositions[1]._mean_feature_vector(),
            np.ndarray,
        )
        # Test that the feature vector function works

    def test_feature_vector(self):
        """Test the feature_vector function."""
        assert isinstance(
            self.valid_magpie_compositions[0].feature_vector(stats=self.stats),
            np.ndarray,
        )
        assert len(
            self.valid_magpie_compositions[0].feature_vector(stats=self.stats),
        ) == self.valid_magpie_compositions[0].embedding.dim * len(self.stats)
        # Test that the feature vector function works with a single stat
        assert isinstance(
            self.valid_magpie_compositions[0].feature_vector(stats="mean"),
            np.ndarray,
        )
        with pytest.raises(ValueError):
            self.valid_magpie_compositions[0].feature_vector(stats="invalid_stat")

    def test_composition_featuriser(self):
        """Test the composition featuriser function."""
        formulas = self.formulas[:3]
        formula_df = pd.DataFrame(formulas, columns=["formula"])
        assert isinstance(composition.composition_featuriser(formula_df), pd.DataFrame)
        assert composition.composition_featuriser(formula_df).shape == (3, 23)
        assert isinstance(composition.composition_featuriser(formulas), list)
        assert len(composition.composition_featuriser(formulas)) == 3

    def test_composition_distance(self):
        """Test the distance method of the CompositionalEmbedding class."""
        assert isinstance(
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[1],
            ),
            float,
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[1],
            )
            > 0
        )

        assert self.valid_magpie_compositions[0].distance(
            self.valid_magpie_compositions[1],
        ) == self.valid_magpie_compositions[1].distance(
            self.valid_magpie_compositions[0],
        )

        self.assertAlmostEqual(
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[1],
            ),
            204.65052421,
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[0],
            )
            == 0
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[1],
                stats="mean",
            )
            > 0
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[0],
                stats="mean",
            )
            == 0
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[1],
                stats=["mean", "variance"],
            )
            > 0
        )
        assert (
            self.valid_magpie_compositions[0].distance(
                self.valid_magpie_compositions[0],
                stats=["mean", "variance"],
            )
            == 0
        )

        assert self.valid_magpie_compositions[0].distance(
            self.formulas[1],
            stats=["mean", "variance"],
            distance_metric="cosine_distance",
        ) == self.valid_magpie_compositions[1].distance(
            self.valid_magpie_compositions[0],
            stats=["mean", "variance"],
            distance_metric="cosine_distance",
        )

        with pytest.raises(TypeError):
            self.valid_magpie_compositions[0].distance(
                5,
                stats=["mean", "variance"],
            )
        with pytest.raises(TypeError):
            self.valid_magpie_compositions[0].distance(
                composition.CompositionalEmbedding(formula="Fe2O3", embedding="skipatom"),
                stats="mean",
            )


class TestSpeciesCompositionalEmbedding(unittest.TestCase):
    """Test the SpeciesCompositionalEmbedding class."""

    def setUp(self):
        """Set up the test compositions."""
        self.compositions = [
            {"Fe2+": 1, "Fe3+": 2, "O2-": 4},
            {"Li+": 7, "La3+": 3, "Zr4+": 1, "O2-": 12},
            {"Cs+": 1, "Pb2+": 1, "I-": 3},
            {"Pb2+": 1, "Pb4+": 1, "O2-": 3},
        ]

        self.valid_skipspecies_compositions = [
            composition.SpeciesCompositionalEmbedding(formula_dict=comp, embedding="skipspecies")
            for comp in self.compositions
        ]

        self.stats = [
            "mean",
            "variance",
            "minpool",
            "maxpool",
            "sum",
            "range",
            "geometric_mean",
            "harmonic_mean",
        ]

    def test_SpeciesCompositionalEmbedding_attributes(self):
        """Test the SpeciesCompositionalEmbedding class."""
        Fe3O4_skipspecies = self.valid_skipspecies_compositions[0]
        assert isinstance(Fe3O4_skipspecies.embedding, core.SpeciesEmbedding)
        assert Fe3O4_skipspecies.composition == self.compositions[0]
        assert Fe3O4_skipspecies.num_atoms == 7
        assert Fe3O4_skipspecies.embedding.dim == 200
        assert Fe3O4_skipspecies.embedding_name == "skipspecies"
        assert Fe3O4_skipspecies.formula_pretty == "Fe3O4"

    def test_feature_vector(self):
        """Test the feature_vector function."""
        assert isinstance(
            self.valid_skipspecies_compositions[0].feature_vector(stats=self.stats),
            np.ndarray,
        )
        assert len(
            self.valid_skipspecies_compositions[0].feature_vector(stats=self.stats),
        ) == self.valid_skipspecies_compositions[0].embedding.dim * len(self.stats)
        # Test that the feature vector function works with a single stat
        assert isinstance(
            self.valid_skipspecies_compositions[0].feature_vector(stats="mean"),
            np.ndarray,
        )

    def test_species_composition_featuriser(self):
        """Test the composition featuriser function."""
        featurised_comps = composition.species_composition_featuriser(self.compositions)
        featurised_comps_df = composition.species_composition_featuriser(self.compositions, to_dataframe=True)

        assert isinstance(
            featurised_comps,
            list,
        )
        assert len(composition.species_composition_featuriser(self.compositions)) == 4
        assert isinstance(
            featurised_comps_df,
            pd.DataFrame,
        )
        assert featurised_comps_df.shape == (4, 202)

    def test_composition_distance(self):
        """Test the distance method of the SpeciesCompositionalEmbedding class."""
        assert isinstance(
            self.valid_skipspecies_compositions[0].distance(
                self.valid_skipspecies_compositions[1],
            ),
            float,
        )
        assert (
            self.valid_skipspecies_compositions[0].distance(
                self.valid_skipspecies_compositions[1],
            )
            > 0
        )

        assert self.valid_skipspecies_compositions[0].distance(
            self.valid_skipspecies_compositions[1],
        ) == self.valid_skipspecies_compositions[1].distance(
            self.valid_skipspecies_compositions[0],
        )

        self.assertAlmostEqual(
            self.valid_skipspecies_compositions[0].distance(
                self.valid_skipspecies_compositions[1],
            ),
            0.9718074189640816,
        )

        assert (
            self.valid_skipspecies_compositions[0].distance(
                self.valid_skipspecies_compositions[0],
            )
            == 0
        )

        assert (
            self.valid_skipspecies_compositions[0].distance(
                self.valid_skipspecies_compositions[1],
                stats="mean",
            )
            > 0
        )
