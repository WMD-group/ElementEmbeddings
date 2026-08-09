"""Test the compare module."""

from __future__ import annotations

import unittest

import pytest

from elementembeddings.compare import (
    embedding_similarity,
    frobenius_distance,
    kl_divergence,
    mantel_test,
    pairwise_embedding_comparison,
)
from elementembeddings.core import Embedding


class TestCompare(unittest.TestCase):
    """Test the compare module against built-in embedding schemes."""

    @classmethod
    def setUpClass(cls):
        cls.magpie = Embedding.load_data("magpie")
        cls.mat2vec = Embedding.load_data("mat2vec")
        cls.megnet = Embedding.load_data("megnet16")

    def test_embedding_similarity_self(self):
        r = embedding_similarity(self.magpie, self.magpie)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_embedding_similarity_methods(self):
        for comparison in ("pearson", "spearman", "kendall"):
            r = embedding_similarity(self.magpie, self.mat2vec, comparison=comparison)
            assert -1.0 <= r <= 1.0

    def test_embedding_similarity_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown comparison method"):
            embedding_similarity(self.magpie, self.mat2vec, comparison="bogus")

    def test_mantel_test_self(self):
        r, p = mantel_test(self.magpie, self.magpie, n_permutations=49)
        assert r == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < p <= 1.0

    def test_mantel_test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown Mantel method"):
            mantel_test(self.magpie, self.mat2vec, method="kendall")

    def test_kl_divergence_non_negative(self):
        d = kl_divergence(self.magpie, self.mat2vec)
        assert d >= 0.0

    def test_kl_divergence_self_zero(self):
        d = kl_divergence(self.magpie, self.magpie)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_frobenius_distance_self_zero(self):
        d = frobenius_distance(self.magpie, self.magpie)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_pairwise_embedding_comparison_shape(self):
        embs = {"magpie": self.magpie, "mat2vec": self.mat2vec, "megnet": self.megnet}
        df = pairwise_embedding_comparison(embs)
        assert df.shape == (3, 3)
        assert list(df.index) == list(df.columns) == ["magpie", "mat2vec", "megnet"]
        # Diagonal should be ~1
        for name in df.index:
            assert df.loc[name, name] == pytest.approx(1.0, abs=1e-6)
        # Symmetric
        assert df.loc["magpie", "mat2vec"] == pytest.approx(df.loc["mat2vec", "magpie"])
