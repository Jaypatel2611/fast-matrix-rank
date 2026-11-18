# fast-matrix-rank

Fast numerical rank for large **sparse, square, full-rank** matrices, with a safe fallback to SVD for everything else.

`numpy.linalg.matrix_rank` needs a dense SVD: O(N³) time and O(N²) memory. For a banded or sparse matrix with N = 100,000 that is impossible (80 GB just to hold it). This package checks rank with a sparse LU factorization (`scipy.sparse.linalg.splu`) instead, and falls back to SVD whenever the LU result cannot be trusted.

## Disclosure

This project was partially written with the assistance of Claude, an AI assistant. The code, tests and benchmark script were developed with its help. I reviewed and ran them, and the benchmark numbers in this README were measured on my own machine.

## Install

```bash
pip install fast-matrix-rank
```

To install from source:

```bash
git clone https://github.com/Jaypatel2611/fast-matrix-rank.git
cd fast-matrix-rank
pip install -e .
```

## Usage

```python
import numpy as np
import scipy.sparse as sp
from fast_matrix_rank import hybrid_matrix_rank, nullity

n = 100_000
A = sp.diags([np.ones(n - 1), 4 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc")

hybrid_matrix_rank(A)   # 100000, in tens of milliseconds
nullity(A)              # 0
```

By default the function may convert a sparse matrix to a dense array when sparse LU cannot decide the rank. To forbid that, pass `allow_dense=False`. The function then raises `RankNotDeterminedError` instead of densifying:

```python
from fast_matrix_rank import hybrid_matrix_rank, RankNotDeterminedError

try:
    r = hybrid_matrix_rank(A, allow_dense=False)
except RankNotDeterminedError:
    ...  # the matrix is singular, badly scaled, or not square
```

Dense NumPy arrays, `scipy.sparse` matrices and the newer sparse arrays (`csr_array` etc.) are all accepted. Real, integer and complex data work. NaN or Inf raises `ValueError`.

## How it chooses a method

| Input | Method |
|---|---|
| Square, sparse (density ≤ 15 %), full rank | Sparse LU |
| Small dense (max dimension ≤ 500) | Gaussian elimination (see note below) |
| Large dense, rectangular, rank-deficient, ill-conditioned | LAPACK SVD |

You can force a method with `method="gauss"` or `method="svd"`. With `allow_dense=False`, any case that would need a dense conversion raises `RankNotDeterminedError`. Dense NumPy input is not affected by this option.

## Benchmarks

Measured with `python benchmark.py` on Windows, AMD64 (AuthenticAMD Family 25), Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1. Your numbers will differ, so run it yourself.

Sparse tridiagonal, full rank:

| N | This package | Dense SVD (`numpy.linalg.matrix_rank`) |
|---|---|---|
| 1,000 | 1.61 ms | 0.16 s |
| 2,000 | 1.72 ms | 1.23 s |
| 4,000 | 2.70 ms | 14.00 s |
| 10,000 | 9.3 ms | not run (0.8 GB matrix) |
| 100,000 | 90 ms | not run (80 GB matrix) |

At small sizes the gain is modest (about 100× at N = 1,000) because dense SVD is still fast there.

## Related SciPy functions

SciPy has no exact numerical rank for sparse matrices. These existing functions are related:

- `scipy.linalg.interpolative.estimate_rank` estimates the rank to a given relative precision using randomized methods. It accepts a `numpy.ndarray` or a `LinearOperator` (so sparse matrices can be used through a wrapper). It returns an estimate, not an exact rank.
- `scipy.sparse.csgraph.structural_rank` works on sparse matrices but only looks at the sparsity pattern, not the values. It is an upper bound on the numerical rank.

This package checks the actual values of a square sparse matrix with sparse LU, and can fall back to SVD (or raise an error with `allow_dense=False`).

## Limitations

Please read these before relying on the speedup.

- **Only full-rank sparse matrices are fast.** If the LU factorization finds the matrix singular or badly scaled, the code converts to a dense array and runs SVD (or raises `RankNotDeterminedError` if `allow_dense=False`). A rank-deficient sparse matrix therefore costs about the same as plain `matrix_rank` (1.34 s vs 1.20 s at N = 2000 in my test).
- **Rectangular sparse matrices are converted to dense**, or raise an error with `allow_dense=False`.
- **LU is not rank-revealing.** A clean-looking `U` diagonal does not prove that a matrix is nonsingular. In my tests, about 15 % of random singular tridiagonal matrices (one duplicated row, N = 500) had no small pivot and an early version wrongly reported full rank. The package therefore also estimates the condition number (`scipy.sparse.linalg.onenormest` on the inverse, using the LU factors) and trusts the LU result only when that estimate is safely below 1/tolerance. The estimate is not a proof, so SVD remains the reference for borderline matrices.
- **The O(N) behaviour applies to banded or low-fill matrices.** General sparse LU can suffer heavy fill-in.
- **The Gaussian elimination route is not faster than SVD.** In my benchmark it was 2–7× slower than `numpy.linalg.matrix_rank` for N from 50 to 1000. It is kept for transparency and as an option (`method="gauss"`), not as an optimization.

## Tests

```bash
pip install pytest
pytest
```

## Contributing

Issues and pull requests are welcome, especially benchmarks on other hardware and examples of matrices where the LU path gives the wrong rank. See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
