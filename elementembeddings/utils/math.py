"""Math functions for the AtomicEmbeddings package."""

import operator
from typing import List, Union


def dot(a: List[Union[int, float]], b: List[Union[int, float]]) -> Union[int, float]:
    """Dot product of two vectors."""
    return sum(map(operator.mul, a, b))


def cosine_similarity(
    a: List[Union[int, float]], b: List[Union[int, float]]
) -> Union[int, float]:
    """Cosine similarity of two vectors."""
    return dot(a, b) / ((dot(a, a) ** 0.5) * (dot(b, b) ** 0.5))


def cosine_distance(
    a: List[Union[int, float]], b: List[Union[int, float]]
) -> Union[int, float]:
    """Cosine distance of two vectors."""
    return 1 - cosine_similarity(a, b)
