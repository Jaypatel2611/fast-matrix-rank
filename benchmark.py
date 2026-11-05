"""Reproducible benchmark. Run: python benchmark.py"""
import platform
import time

import numpy as np
import scipy
import scipy.sparse as sp

from fast_matrix_rank import hybrid_matrix_rank, vectorized_gauss_rank


def best_of(fn, repeat=3):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return out, min(times)


rng = np.random.default_rng(42)
print(f"Python {platform.python_version()} | NumPy {np.__version__} | "
      f"SciPy {scipy.__version__} | {platform.processor() or platform.machine()}")

print("\nDense random: vectorized Gauss vs numpy.linalg.matrix_rank (best of 3)")
print(f"{'N':>6} {'gauss ms':>10} {'svd ms':>10} {'same rank':>10}")
for n in (50, 100, 200, 500, 1000):
    A = rng.standard_normal((n, n))
    rg, tg = best_of(lambda: vectorized_gauss_rank(A))
    rs, ts = best_of(lambda: np.linalg.matrix_rank(A))
    print(f"{n:>6} {tg*1e3:>10.2f} {ts*1e3:>10.2f} {str(rg == rs):>10}")

print("\nSparse tridiagonal, full rank: hybrid vs dense SVD")
print(f"{'N':>7} {'hybrid ms':>10} {'dense svd s':>12} {'speedup':>9} {'same rank':>10}")
for n in (1000, 2000, 4000):
    d = rng.standard_normal(n) + 4.0
    o = rng.standard_normal(n - 1)
    A = sp.diags([o, d, o], [-1, 0, 1], format="csc")
    rh, th = best_of(lambda: hybrid_matrix_rank(A))
    rd, td = best_of(lambda: np.linalg.matrix_rank(A.toarray()), repeat=1)
    print(f"{n:>7} {th*1e3:>10.2f} {td:>12.2f} {td/th:>8.0f}x {str(rh == rd):>10}")

for n in (10_000, 100_000):
    d = rng.standard_normal(n) + 4.0
    o = rng.standard_normal(n - 1)
    A = sp.diags([o, d, o], [-1, 0, 1], format="csc")
    r, t = best_of(lambda: hybrid_matrix_rank(A))
    print(f"{n:>7} {t*1e3:>10.2f} ms   (dense SVD not run: needs {n*n*8/1e9:.1f} GB)")

print("\nSparse tridiagonal, rank-deficient (one duplicated row): NO speedup expected")
n = 2000
d = rng.standard_normal(n) + 4.0
o = rng.standard_normal(n - 1)
A = sp.diags([o, d, o], [-1, 0, 1], format="lil")
A[5, :] = A[4, :]
A = A.tocsc()
rh, th = best_of(lambda: hybrid_matrix_rank(A), repeat=1)
rd, td = best_of(lambda: np.linalg.matrix_rank(A.toarray()), repeat=1)
print(f"  hybrid {th:.2f}s rank={rh} | dense svd {td:.2f}s rank={rd}")
