"""TOOL: F2 linear algebra on int bitmasks (rows as Python ints)."""


def rref(rows):
    """Reduced row echelon form. Returns dict pivot_col -> row."""
    pivots = {}
    for row in rows:
        r = row
        while r:
            c = r.bit_length() - 1
            if c in pivots:
                r ^= pivots[c]
            else:
                pivots[c] = r
                break
    for c in sorted(pivots, reverse=True):
        for c2 in pivots:
            if c2 != c and (pivots[c2] >> c) & 1:
                pivots[c2] ^= pivots[c]
    return pivots


def nullspace(rows, nvars):
    """Basis of {v : for all rows r, parity(r & v) = 0}."""
    pivots = rref(rows)
    basis = []
    for v in range(nvars):
        if v in pivots:
            continue
        vec = 1 << v
        for c, row in pivots.items():
            if (row >> v) & 1:
                vec |= 1 << c
        basis.append(vec)
    return basis


def in_span(vec, rows):
    r = vec
    piv = rref(rows)
    while r:
        c = r.bit_length() - 1
        if c not in piv:
            return False
        r ^= piv[c]
    return True


def rank(rows):
    return len(rref(rows))


def reduce_mod(vec, pivots):
    """Fully reduce vec against RREF pivot rows; the remainder has no pivot
    columns set. (Single pass in any order is exact: after full RREF, no
    pivot row contains another row's pivot column.)"""
    r = vec
    for c, row in pivots.items():
        if (r >> c) & 1:
            r ^= row
    return r


def quotient_basis(space_rows, sub_rows):
    """Representatives of space / span(sub): reduce each space row against
    the subspace and keep the independent nonzero remainders."""
    piv_sub = rref(sub_rows)
    piv_out = {}
    reps = []
    for v in space_rows:
        r = reduce_mod(v, piv_sub)
        # then reduce against already-kept representatives
        while r:
            c = r.bit_length() - 1
            if c in piv_out:
                r ^= piv_out[c]
            else:
                piv_out[c] = r
                reps.append(r)
                break
    return reps
