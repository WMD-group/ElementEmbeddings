"""Math functions for the AtomicEmbeddings package."""

from __future__ import annotations

import operator


def dot(a: list[int | float], b: list[int | float]) -> int | float:
    """Dot product of two vectors."""
    return sum(map(operator.mul, a, b))


def cosine_similarity(
    a: list[int | float],
    b: list[int | float],
) -> int | float:
    """Cosine similarity of two vectors."""
    return dot(a, b) / ((dot(a, a) ** 0.5) * (dot(b, b) ** 0.5))


def cosine_distance(
    a: list[int | float],
    b: list[int | float],
) -> int | float:
    """Cosine distance of two vectors."""
    return 1 - cosine_similarity(a, b)
