"""Test the modules in the utils package."""

from __future__ import annotations

import unittest
from typing import ClassVar

import numpy as np

from elementembeddings.utils import io, math, species


# ------------ math.py functions ------------
class TestMath(unittest.TestCase):
    """Test the math module."""

    a: ClassVar = [1, 2, 3, 4, 5]
    b: ClassVar = [1, 2, 3, 4, 5]
    c: ClassVar = [-1, 1 / 2, -1 / 3, 1 / 4, 0]

    def test_dot(self):
        """Test the dot function."""
        assert math.dot(self.a, self.b) == 55
        assert math.dot(self.a, self.c) == 0

    def test_cosine_similarity(self):
        """Test the cosine_similarity function."""
        assert math.cosine_similarity(self.a, self.b) == 1
        assert math.cosine_similarity(self.a, self.c) == 0

    def test_cosine_distance(self):
        """Test the cosine_distance function."""
        assert math.cosine_distance(self.a, self.b) == 0
        assert math.cosine_distance(self.a, self.c) == 1


class TestIO(unittest.TestCase):
    """Test the io module."""

    def test_numpy_encoder(self):
        """Test the NumpyEncoder class."""
        encoder = io.NumpyEncoder()
        assert encoder.default(np.array([1, 2, 3])) == [1, 2, 3]
        assert encoder.default(np.array([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


class TestSpecies(unittest.TestCase):
    """Test the species module."""

    def test_parse_species(self):
        """Test the parse_species function."""
        assert species.parse_species("Fe") == ("Fe", 0)
        assert species.parse_species("Fe0") == ("Fe", 0)
        assert species.parse_species("Fe0+") == ("Fe", 0)
        assert species.parse_species("Fe0-") == ("Fe", 0)
        assert species.parse_species("Fe1+") == ("Fe", 1)
        assert species.parse_species("Fe1-") == ("Fe", -1)
        assert species.parse_species("Fe+") == ("Fe", 1)
        assert species.parse_species("Fe-") == ("Fe", -1)
        assert species.parse_species("Fe2.5+") == ("Fe", 2.5)
        assert species.parse_species("Fe2.5-") == ("Fe", -2.5)
        assert species.parse_species("Fe2.555+") == ("Fe", 2.555)
