import numpy as np
import pytest
import scipy.sparse as sp

from fast_matrix_rank import (
    RankNotDeterminedError,
    hybrid_matrix_rank,
    nullity,
    vectorized_gauss_rank,
)

rng = np.random.default_rng(0)


def tridiag(n, dup_row=False):
    d = rng.standard_normal(n) + 4.0
    o = rng.standard_normal(n - 1)
    A = sp.diags([o, d, o], [-1, 0, 1], format="lil")
    if dup_row:
        A[5, :] = A[4, :]
    return A.tocsc()


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (50, 30), (30, 50), (100, 100)])
def test_dense_random_matches_numpy(shape):
    A = rng.standard_normal(shape)
    assert vectorized_gauss_rank(A) == np.linalg.matrix_rank(A)
    assert hybrid_matrix_rank(A) == np.linalg.matrix_rank(A)


def test_low_rank_dense():
    A = rng.standard_normal((60, 7)) @ rng.standard_normal((7, 80))
    assert hybrid_matrix_rank(A) == 7
    assert nullity(A) == 80 - 7


def test_complex_keeps_imaginary_part():
    A = np.array([[1j, 0], [0, 1j]])
    assert vectorized_gauss_rank(A) == 2


def test_zero_and_empty():
    assert hybrid_matrix_rank(np.zeros((4, 4))) == 0
    assert hybrid_matrix_rank(np.zeros((0, 3))) == 0
    assert hybrid_matrix_rank(sp.csc_matrix((5, 5))) == 0


def test_integer_and_list_input():
    assert hybrid_matrix_rank([[1, 2], [2, 4]]) == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_inf_rejected(bad):
    with pytest.raises(ValueError):
        hybrid_matrix_rank(np.array([[1.0, bad], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        hybrid_matrix_rank(sp.csc_matrix(np.array([[1.0, bad], [2.0, 3.0]])))


def test_not_2d():
    with pytest.raises(ValueError):
        hybrid_matrix_rank(np.zeros(3))


def test_sparse_full_rank():
    assert hybrid_matrix_rank(tridiag(3000)) == 3000


def test_sparse_exactly_singular_does_not_crash():
    A = tridiag(1500, dup_row=True)
    assert hybrid_matrix_rank(A) == np.linalg.matrix_rank(A.toarray()) == 1499


def test_sparse_ill_conditioned_falls_back_to_svd():
    d = np.ones(800)
    d[400] = 1e-14
    A = sp.diags(d).tocsc()
    assert hybrid_matrix_rank(A) == np.linalg.matrix_rank(A.toarray()) == 799


@pytest.mark.parametrize("ctor", [sp.csr_matrix, sp.csr_array, sp.coo_array, sp.csc_array])
def test_all_sparse_containers(ctor):
    A = ctor(tridiag(1000))
    assert hybrid_matrix_rank(A) == 1000


def test_rectangular_sparse():
    A = sp.random(200, 300, density=0.02, random_state=1, format="csr")
    assert hybrid_matrix_rank(A) == np.linalg.matrix_rank(A.toarray())


def test_method_overrides():
    A = rng.standard_normal((40, 40))
    for m in ("gauss", "svd", "auto"):
        assert hybrid_matrix_rank(A, method=m) == 40
    with pytest.raises(ValueError):
        hybrid_matrix_rank(A, method="nope")


def test_no_dense_full_rank_sparse_still_works():
    assert hybrid_matrix_rank(tridiag(2000), allow_dense=False) == 2000
    assert nullity(tridiag(2000), allow_dense=False) == 0


def test_no_dense_singular_raises():
    with pytest.raises(RankNotDeterminedError):
        hybrid_matrix_rank(tridiag(500, dup_row=True), allow_dense=False)


def test_no_dense_ill_conditioned_raises():
    d = np.ones(300)
    d[100] = 1e-14
    with pytest.raises(RankNotDeterminedError):
        hybrid_matrix_rank(sp.diags(d).tocsc(), allow_dense=False)


def test_no_dense_rectangular_sparse_raises():
    A = sp.random(20, 30, density=0.05, random_state=1, format="csr")
    with pytest.raises(RankNotDeterminedError):
        hybrid_matrix_rank(A, allow_dense=False)


@pytest.mark.parametrize("method", ["gauss", "svd"])
def test_no_dense_forced_method_on_sparse_raises(method):
    with pytest.raises(RankNotDeterminedError):
        hybrid_matrix_rank(tridiag(50), method=method, allow_dense=False)


def test_no_dense_does_not_affect_dense_input():
    A = rng.standard_normal((30, 30))
    assert hybrid_matrix_rank(A, allow_dense=False) == 30
