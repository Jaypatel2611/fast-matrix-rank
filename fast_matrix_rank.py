"""Adaptive matrix rank estimation.

Routes a matrix to the cheapest reliable method:

* square, sparse, full-rank          -> sparse LU (``scipy.sparse.linalg.splu``)
* small dense                        -> vectorized Gaussian elimination
* everything else / any doubt        -> LAPACK SVD (``numpy.linalg.matrix_rank``)

Limits (by design):

* The sparse fast path only helps when the matrix is full rank. A
  rank-deficient or ill-conditioned sparse matrix falls back to *dense* SVD,
  so it costs O(N^3) time and O(N^2) memory.
* Rectangular sparse matrices are densified.
* The Gaussian route uses partial pivoting, which is not rank-revealing in
  the worst case. SVD stays the reference for ill-conditioned input.

License: MIT
"""

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["vectorized_gauss_rank", "hybrid_matrix_rank", "nullity"]

# If min|U_ii| / max|U_ii| is below this, the LU rank is not trusted.
_LU_RATIO_TRUST = 1e-8


def _default_tol(shape, scale: float, rtol: Optional[float]) -> float:
    eps = float(np.finfo(float).eps)
    return (rtol if rtol is not None else max(shape) * eps) * scale


def _check_finite(values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("Matrix contains NaN or Inf.")


def vectorized_gauss_rank(A: np.ndarray, rtol: Optional[float] = None) -> int:
    """Rank via Gaussian elimination with partial pivoting.

    Each elimination step is one NumPy outer-product update.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"Input must be 2D, got {A.ndim}D.")
    m, n = A.shape
    if m == 0 or n == 0:
        return 0
    dtype = np.complex128 if np.iscomplexobj(A) else np.float64
    X = np.array(A, dtype=dtype, copy=True)
    _check_finite(X)

    a_max = float(np.abs(X).max())
    if a_max == 0.0:
        return 0
    tol = _default_tol((m, n), a_max, rtol)

    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break
        col_abs = np.abs(X[pivot_row:, col])
        k = int(np.argmax(col_abs))
        if col_abs[k] <= tol:
            continue
        max_row = pivot_row + k
        if max_row != pivot_row:
            X[[pivot_row, max_row]] = X[[max_row, pivot_row]]
        factors = X[pivot_row + 1:, col] / X[pivot_row, col]
        X[pivot_row + 1:, col:] -= factors[:, None] * X[pivot_row, col:]
        pivot_row += 1
    return pivot_row


def _svd_rank(A, rtol: Optional[float]) -> int:
    """NumPy's own default tolerance (relative to the largest singular value)."""
    dense = A.toarray() if sp.issparse(A) else np.asarray(A)
    if dense.size == 0:
        return 0
    _check_finite(dense)
    if rtol is None:
        return int(np.linalg.matrix_rank(dense))
    s = np.linalg.svd(dense, compute_uv=False)
    return int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0


def hybrid_matrix_rank(
    A,
    rtol: Optional[float] = None,
    density_threshold: float = 0.15,
    dense_gauss_cutoff: int = 500,
    method: str = "auto",
) -> int:
    """Numerical rank of a dense or sparse matrix.

    ``method``: "auto", "gauss", "svd". ``rtol`` is relative to the largest
    singular value on the SVD route and to the largest absolute entry on the
    Gauss/LU routes.
    """
    if not sp.issparse(A):
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"Input must be 2D, got {A.ndim}D.")
    if 0 in A.shape:
        return 0
    m, n = A.shape

    if method == "svd":
        return _svd_rank(A, rtol)
    if method == "gauss":
        return vectorized_gauss_rank(A.toarray() if sp.issparse(A) else A, rtol)
    if method != "auto":
        raise ValueError(f"Unknown method {method!r}.")

    data = A.data if sp.issparse(A) else A
    _check_finite(data)

    if sp.issparse(A):
        density = A.nnz / float(m * n)
    else:
        density = np.count_nonzero(A) / float(m * n)

    # Dense, rectangular or tiny -> no sparse LU.
    if density > density_threshold or m != n or n < 2:
        if max(m, n) <= dense_gauss_cutoff:
            return vectorized_gauss_rank(
                A.toarray() if sp.issparse(A) else A, rtol
            )
        return _svd_rank(A, rtol)

    # Square and sparse: try sparse LU.
    A_csc = sp.csc_matrix(A)
    if A_csc.nnz == 0:
        return 0
    try:
        lu = spla.splu(A_csc, diag_pivot_thresh=1.0)
    except RuntimeError:  # "Factor is exactly singular"
        return _svd_rank(A_csc, rtol)

    u_diag = np.abs(lu.U.diagonal())
    u_max = float(u_diag.max())
    # Trust LU only if it says full rank AND the pivots are well scaled.
    if u_max > 0 and float(u_diag.min()) / u_max > _LU_RATIO_TRUST:
        tol = _default_tol((m, n), float(np.abs(A_csc.data).max()), rtol)
        if float(u_diag.min()) > tol:
            return n
    return _svd_rank(A_csc, rtol)


def nullity(A, rtol: Optional[float] = None) -> int:
    """Nullity = columns - rank."""
    if not sp.issparse(A):
        A = np.asarray(A)
    return A.shape[1] - hybrid_matrix_rank(A, rtol=rtol)
