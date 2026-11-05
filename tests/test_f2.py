"""F2 linear algebra invariants."""

import random

from f2 import in_span, nullspace, quotient_basis, rank, reduce_mod, rref


def random_rows(n_rows, n_cols, seed):
    rng = random.Random(seed)
    return [rng.getrandbits(n_cols) for _ in range(n_rows)]


def test_rank_nullity():
    for seed in range(5):
        rows = random_rows(12, 20, seed)
        assert rank(rows) + len(nullspace(rows, 20)) == 20


def test_nullspace_annihilates():
    rows = random_rows(10, 16, 42)
    for v in nullspace(rows, 16):
        for r in rows:
            assert bin(r & v).count("1") % 2 == 0


def test_in_span():
    rows = random_rows(8, 16, 7)
    assert in_span(rows[0] ^ rows[3] ^ rows[5], rows)
    # membership must agree with the rank criterion on random vectors
    for seed in range(10):
        v = random.Random(100 + seed).getrandbits(16)
        assert in_span(v, rows) == (rank(rows + [v]) == rank(rows))


def test_reduce_mod_clears_all_pivots():
    rows = random_rows(9, 18, 3)
    piv = rref(rows)
    for seed in range(10):
        v = random.Random(seed).getrandbits(18)
        r = reduce_mod(v, piv)
        assert all(not (r >> c) & 1 for c in piv)


def test_quotient_dimension():
    # quotient of a space by a subspace of itself: dim = difference
    sub = random_rows(6, 24, 11)
    space = sub + random_rows(5, 24, 12)
    q = quotient_basis(space, sub)
    assert len(q) == rank(space + sub) - rank(sub)
