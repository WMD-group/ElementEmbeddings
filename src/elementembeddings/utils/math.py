"""Math functions for the AtomicEmbeddings package."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def dot(a: Iterable[int | float], b: Iterable[int | float]) -> float:
    """Dot product of two vectors."""
    return float(sum(map(operator.mul, a, b)))


def cosine_similarity(
    a: Iterable[int | float],
    b: Iterable[int | float],
) -> float:
    """Cosine similarity of two vectors."""
    return dot(a, b) / ((dot(a, a) ** 0.5) * (dot(b, b) ** 0.5))


def cosine_distance(
    a: Iterable[int | float],
    b: Iterable[int | float],
) -> float:
    """Cosine distance of two vectors."""
    return 1 - cosine_similarity(a, b)
