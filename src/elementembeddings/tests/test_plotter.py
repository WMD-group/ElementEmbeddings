"""Test the plotter module."""

import unittest

import matplotlib.pyplot as plt

from elementembeddings.core import Embedding
from elementembeddings.plotter import dimension_plotter, heatmap_plotter


class heatmapTest(unittest.TestCase):
    """Test the heatmap_plotter function."""

    def test_heatmap_plotter(self):
        """Test that the heatmap_plotter function works."""
        # Load the data
        skipatom = Embedding.load_data("skipatom")
        # Get the embeddings
        skipatom_cos_plot = heatmap_plotter(
            skipatom,
            metric="cosine_similarity",
        )
        assert isinstance(skipatom_cos_plot, plt.Axes)


class dimensionTest(unittest.TestCase):
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
