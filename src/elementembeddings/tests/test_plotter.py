"""Test the plotter module."""

import unittest

import matplotlib.pyplot as plt

from elementembeddings.core import Embedding
from elementembeddings.plotter import dimension_plotter, heatmap_plotter


class HeatmapTest(unittest.TestCase):
    """Test the heatmap_plotter function."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        cls.test_skipatom = Embedding.load_data("skipatom")

    def test_heatmap_plotter(self):
        """Test that the heatmap_plotter function works."""
        # Get the embeddings
        skipatom_cos_plot = heatmap_plotter(
            self.test_skipatom,
            metric="cosine_similarity",
        )
        assert isinstance(skipatom_cos_plot, plt.Axes)
        skipatom_euc_plot = heatmap_plotter(
            self.test_skipatom, metric="euclidean", show_axislabels=False
        )
        assert isinstance(skipatom_euc_plot, plt.Axes)


class DimensionTest(unittest.TestCase):
    """Test the dimension_plotter function."""

    def test_dimension_plotter(self):
        """Test that the dimension_plotter function works."""
        # Load the data
        skipatom = Embedding.load_data("skipatom")
        # Get the embeddings
        skipatom_pca_plot = dimension_plotter(
            skipatom, n_components=2, reducer="pca", adjusttext=True
        )
        assert isinstance(skipatom_pca_plot, plt.Axes)
