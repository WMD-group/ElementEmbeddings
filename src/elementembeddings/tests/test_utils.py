"""Test the modules in the utils package."""
import unittest

from elementembeddings import utils


# ------------ math.py functions ------------
class TestMath(unittest.TestCase):
    """Test the math module."""

    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    c = [-1, 1 / 2, -1 / 3, 1 / 4, 0]

    def test_dot(self):
        """Test the dot function."""
        assert utils.math.dot(self.a, self.b) == 55
        assert utils.math.dot(self.a, self.c) == 0

    def test_cosine_similarity(self):
        """Test the cosine_similarity function."""
        assert utils.math.cosine_similarity(self.a, self.b) == 1
        assert utils.math.cosine_similarity(self.a, self.c) == 0

    def test_cosine_distance(self):
        """Test the cosine_distance function."""
        assert utils.math.cosine_distance(self.a, self.b) == 0
        assert utils.math.cosine_distance(self.a, self.c) == 1
