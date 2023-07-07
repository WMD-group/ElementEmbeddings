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

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        cls.test_skipatom = Embedding.load_data("skipatom")

    def test_dimension_2d_plotter(self):
        """Test that the dimension_plotter function works."""
        skipatom_pca_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="pca", adjusttext=False
        )
        assert isinstance(skipatom_pca_plot, plt.Axes)
        skipatom_tsne_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="tsne", adjusttext=False
        )
        assert isinstance(skipatom_tsne_plot, plt.Axes)
        skipatom_umap_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="umap", adjusttext=True
        )
        assert isinstance(skipatom_umap_plot, plt.Axes)

        self.assertRaises(
            ValueError,
            dimension_plotter,
            self.test_skipatom,
            n_components=2,
            reducer="badreducer",
        )

    def test_dimension_2d_plotter_preloaded_reduction(self):
        """Test that the dimension_plotter function works with a preloaded reduction."""
        self.test_skipatom.calculate_PC()
        self.test_skipatom.calculate_tSNE()
        self.test_skipatom.calculate_UMAP()

        skipatom_pca_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="pca", adjusttext=False
        )
        assert isinstance(skipatom_pca_plot, plt.Axes)
        skipatom_tsne_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="tsne", adjusttext=False
        )
        assert isinstance(skipatom_tsne_plot, plt.Axes)
        skipatom_umap_plot = dimension_plotter(
            self.test_skipatom, n_components=2, reducer="umap", adjusttext=False
        )
        assert isinstance(skipatom_umap_plot, plt.Axes)

    def test_dimension_3d_plotter(self):
        """Test that the dimension_plotter function works in 3D."""
        skipatom_3d_pca_plot = dimension_plotter(
            self.test_skipatom, n_components=3, reducer="pca", adjusttext=False
        )
        assert isinstance(skipatom_3d_pca_plot, plt.Axes)

    def test_dimension_Nd_plotter(self):
        """Test that the dimension_plotter function will fail in d>3."""
        self.assertRaises(
            ValueError,
            dimension_plotter,
            self.test_skipatom,
            n_components=4,
            reducer="pca",
        )
