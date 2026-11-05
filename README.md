# fast-matrix-rank

Fast numerical rank for large **sparse, square, full-rank** matrices, with a safe fallback to SVD for everything else.

`numpy.linalg.matrix_rank` needs a dense SVD: O(N³) time and O(N²) memory. For a banded or sparse matrix with N = 100,000 that is impossible (80 GB just to hold it). This package checks rank with a sparse LU factorization (`scipy.sparse.linalg.splu`) instead, and falls back to SVD whenever the LU result cannot be trusted.

## Install

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

Dense NumPy arrays, `scipy.sparse` matrices and the newer sparse arrays (`csr_array` etc.) are all accepted. Real, integer and complex data work. NaN or Inf raises `ValueError`.

## How it chooses a method

| Input | Method |
|---|---|
| Square, sparse (density ≤ 15 %), full rank | Sparse LU |
| Small dense (max dimension ≤ 500) | Gaussian elimination (see note below) |
| Large dense, rectangular, rank-deficient, ill-conditioned | LAPACK SVD |

You can force a method with `method="gauss"` or `method="svd"`.

## Benchmarks

Measured with `python benchmark.py` on Windows, AMD64 (AuthenticAMD Family 25), Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1. Your numbers will differ, so run it yourself.

Sparse tridiagonal, full rank:

| N | This package | Dense SVD (`numpy.linalg.matrix_rank`) |
|---|---|---|
| 1,000 | 0.50 ms | 0.16 s |
| 2,000 | 0.81 ms | 1.38 s |
| 4,000 | 1.31 ms | 15.47 s |
| 10,000 | 3.4 ms | not run (0.8 GB matrix) |
| 100,000 | 38 ms | not run (80 GB matrix) |

## Limitations

Please read these before relying on the speedup.

- **Only full-rank sparse matrices are fast.** If the LU factorization finds the matrix singular or badly scaled, the code converts to a dense array and runs SVD. A rank-deficient sparse matrix therefore costs the same as plain `matrix_rank` (1.31 s vs 1.25 s at N = 2000 in my test).
- **Rectangular sparse matrices are converted to dense.**
- **LU is not rank-revealing in general.** The package trusts the LU result only when the pivots are well scaled, and uses SVD otherwise. SVD remains the reference for ill-conditioned matrices.
- **The O(N) behaviour applies to banded or low-fill matrices.** General sparse LU can suffer heavy fill-in.
- **The Gaussian elimination route is not faster than SVD.** In my benchmark it was 2–6× slower than `numpy.linalg.matrix_rank` for N from 50 to 1000. It is kept for transparency and as an option (`method="gauss"`), not as an optimization.

## Tests

```bash
pip install pytest
pytest
```

## Contributing

Issues and pull requests are welcome, especially benchmarks on other hardware and examples of matrices where the LU path gives the wrong rank.

## License

MIT. See `LICENSE`.
