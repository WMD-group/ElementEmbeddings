"""Compare element similarity structures across different embedding schemes.

This module provides functions to quantitatively compare how different
embedding schemes represent chemical similarity between elements.

Example usage::

    from elementembeddings.core import Embedding
    from elementembeddings.compare import (
        embedding_similarity,
        mantel_test,
        pairwise_embedding_comparison,
    )

    magpie = Embedding.load_data("magpie")
    mat2vec = Embedding.load_data("mat2vec")

    # Pearson correlation between cosine similarity matrices
    r = embedding_similarity(magpie, mat2vec)

    # Mantel test with p-value
    r, p = mantel_test(magpie, mat2vec)

    # Compare all embeddings pairwise
    embeddings = {name: Embedding.load_data(name) for name in ["magpie", "mat2vec", "megnet16"]}
    comparison_df = pairwise_embedding_comparison(embeddings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .core import Embedding


def _get_common_elements(emb1: Embedding, emb2: Embedding) -> list[str]:
    """Return sorted list of elements common to both embeddings."""
    common = sorted(set(emb1.element_list) & set(emb2.element_list))
    if len(common) < 3:
        msg = f"Only {len(common)} common elements between embeddings. Need at least 3."
        raise ValueError(msg)
    return common


def _get_similarity_matrix(
    emb: Embedding,
    elements: list[str],
    metric: str = "cosine_similarity",
) -> np.ndarray:
    """Compute pairwise similarity matrix for given elements.

    Args:
        emb: Embedding instance.
        elements: List of element symbols to include.
        metric: Similarity or distance metric.

    Returns:
        NxN numpy array of pairwise similarities.
    """
    n = len(elements)
    mat = np.zeros((n, n))
    for i, el1 in enumerate(elements):
        for j, el2 in enumerate(elements):
            if i <= j:
                val = emb.compute_correlation_metric(el1, el2, metric=metric)
                mat[i, j] = val
                mat[j, i] = val
    return mat


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Extract upper triangle values (excluding diagonal) as flat array."""
    return mat[np.triu_indices_from(mat, k=1)]


def embedding_similarity(
    emb1: Embedding,
    emb2: Embedding,
    metric: str = "cosine_similarity",
    comparison: str = "pearson",
) -> float:
    """Compare two embeddings by correlating their element similarity matrices.

    Computes the pairwise element similarity matrix for each embedding,
    then correlates the flattened upper triangles.

    Args:
        emb1: First embedding.
        emb2: Second embedding.
        metric: Similarity metric for element pairs (default: cosine_similarity).
        comparison: Correlation method for comparing matrices.
            One of "pearson", "spearman", "kendall".

    Returns:
        Correlation coefficient between the two similarity matrices.
    """
    elements = _get_common_elements(emb1, emb2)
    mat1 = _get_similarity_matrix(emb1, elements, metric)
    mat2 = _get_similarity_matrix(emb2, elements, metric)

    v1 = _upper_triangle(mat1)
    v2 = _upper_triangle(mat2)

    if comparison == "pearson":
        return float(stats.pearsonr(v1, v2).statistic)
    if comparison == "spearman":
        return float(stats.spearmanr(v1, v2).statistic)
    if comparison == "kendall":
        return float(stats.kendalltau(v1, v2).statistic)

    msg = f"Unknown comparison method: {comparison}. Use 'pearson', 'spearman', or 'kendall'."
    raise ValueError(msg)


def mantel_test(
    emb1: Embedding,
    emb2: Embedding,
    metric: str = "cosine_similarity",
    method: str = "pearson",
    n_permutations: int = 999,
) -> tuple[float, float]:
    """Mantel test for correlation between two embedding similarity matrices.

    Permutation-based significance test for the correlation between two
    distance/similarity matrices. Standard in ecology and chemistry for
    comparing pairwise relationship structures.

    Args:
        emb1: First embedding.
        emb2: Second embedding.
        metric: Similarity metric for element pairs.
        method: Correlation method ("pearson" or "spearman").
        n_permutations: Number of permutations for p-value estimation.

    Returns:
        Tuple of (correlation_coefficient, two_sided_p_value).
        The p-value is two-sided: the fraction of permutations whose
        absolute correlation is at least as large as |observed|.
    """
    if method == "pearson":
        corr_fn = stats.pearsonr
    elif method == "spearman":
        corr_fn = stats.spearmanr
    else:
        msg = f"Unknown Mantel method: {method}. Use 'pearson' or 'spearman'."
        raise ValueError(msg)

    elements = _get_common_elements(emb1, emb2)
    mat1 = _get_similarity_matrix(emb1, elements, metric)
    mat2 = _get_similarity_matrix(emb2, elements, metric)

    v1 = _upper_triangle(mat1)
    v2 = _upper_triangle(mat2)

    observed = float(corr_fn(v1, v2).statistic)
    threshold = abs(observed)

    n = len(elements)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        mat2_perm = mat2[np.ix_(perm, perm)]
        v2_perm = _upper_triangle(mat2_perm)
        perm_corr = float(corr_fn(v1, v2_perm).statistic)
        if abs(perm_corr) >= threshold:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def kl_divergence(
    emb1: Embedding,
    emb2: Embedding,
    metric: str = "cosine_similarity",
) -> float:
    """KL divergence between normalised similarity distributions.

    Normalises each embedding's similarity matrix into a probability
    distribution using softmax, then computes the KL divergence
    D_KL(emb1 || emb2).

    Args:
        emb1: First embedding (the "true" distribution).
        emb2: Second embedding (the "approximate" distribution).
        metric: Similarity metric for element pairs.

    Returns:
        KL divergence (non-negative, 0 means identical distributions).
        Note: this is asymmetric — D_KL(A||B) != D_KL(B||A).
    """
    elements = _get_common_elements(emb1, emb2)
    mat1 = _get_similarity_matrix(emb1, elements, metric)
    mat2 = _get_similarity_matrix(emb2, elements, metric)

    v1 = _upper_triangle(mat1)
    v2 = _upper_triangle(mat2)

    # Softmax normalisation to probability distributions
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    p = _softmax(v1)
    q = _softmax(v2)

    # Clip to avoid log(0)
    q = np.clip(q, 1e-12, None)
    p = np.clip(p, 1e-12, None)

    return float(np.sum(p * np.log(p / q)))


def frobenius_distance(
    emb1: Embedding,
    emb2: Embedding,
    metric: str = "cosine_similarity",
    normalise: bool = True,
) -> float:
    """Frobenius norm of the difference between two similarity matrices.

    Args:
        emb1: First embedding.
        emb2: Second embedding.
        metric: Similarity metric for element pairs.
        normalise: If True, normalise by the number of element pairs.

    Returns:
        Frobenius distance between the two similarity matrices.
    """
    elements = _get_common_elements(emb1, emb2)
    mat1 = _get_similarity_matrix(emb1, elements, metric)
    mat2 = _get_similarity_matrix(emb2, elements, metric)

    diff = mat1 - mat2
    frob = float(np.linalg.norm(diff, "fro"))

    if normalise:
        n_pairs = len(elements) * (len(elements) - 1) / 2
        frob /= np.sqrt(n_pairs)

    return frob


def pairwise_embedding_comparison(
    embeddings: dict[str, Embedding],
    metric: str = "cosine_similarity",
    comparison: str = "pearson",
) -> pd.DataFrame:
    """Compare all pairs of embeddings, returning a comparison matrix.

    Args:
        embeddings: Dictionary mapping embedding names to Embedding objects.
        metric: Similarity metric for element pairs.
        comparison: Correlation method for comparing matrices.

    Returns:
        DataFrame with embedding names as index/columns and correlation
        values as entries.
    """
    names = list(embeddings.keys())
    n = len(names)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            r = embedding_similarity(
                embeddings[names[i]],
                embeddings[names[j]],
                metric=metric,
                comparison=comparison,
            )
            mat[i, j] = r
            mat[j, i] = r

    return pd.DataFrame(mat, index=names, columns=names)
