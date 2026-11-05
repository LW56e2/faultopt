"""TOOL: code definitions and schedule structures.

- Rotated surface code construction (Check / build_code) with corner geometry.
- Generic CSS code container (CSS) + concrete codes: surface_css, steane_css.
- Schedule representations: uniform corner-permutation schedules
  (layer_gates, schedule_valid) and generic per-check edge->layer assignments
  (uniform_assignment converts the former into the latter).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Rotated surface code, distance d
# ---------------------------------------------------------------------------
# Data qubits on a d x d grid, id = r*d + c, stim coords (2c+1, 2r+1).
# Plaquette "cells" at (r, c), r, c in 0..d, corners = data qubits at
# (r-1,c-1), (r-1,c), (r,c-1), (r,c) intersected with the grid.
# Checkerboard: cell type X iff (r+c) even. Weight-2 boundary cells kept iff
# on a matching boundary: top/bottom rows host X checks, left/right columns
# host Z checks. => logical Z = top data row, logical X = left data column.

CORNERS = ("NW", "NE", "SW", "SE")
OFFSET = {"NW": (-1, -1), "NE": (-1, 0), "SW": (0, -1), "SE": (0, 0)}


@dataclass(frozen=True)
class Check:
    kind: str          # "Z" or "X"
    r: int             # cell row
    c: int             # cell col
    anc: int           # ancilla qubit id
    data: tuple        # tuple of (corner_name, data_qubit_id)

    def corner_map(self):
        return dict(self.data)


def build_code(d):
    """Return list of Check for the rotated surface code of distance d."""
    checks = []
    anc = d * d
    for r in range(d + 1):
        for c in range(d + 1):
            kind = "X" if (r + c) % 2 == 0 else "Z"
            data = []
            for name in CORNERS:
                dr, dc = OFFSET[name]
                rr, cc = r + dr, c + dc
                if 0 <= rr < d and 0 <= cc < d:
                    data.append((name, rr * d + cc))
            if len(data) < 2:
                continue  # corners of the patch: weight-1, not a stabilizer
            if len(data) == 2:
                if (r == 0 or r == d) and kind != "X":
                    continue
                if (c == 0 or c == d) and kind != "Z":
                    continue
            checks.append(Check(kind, r, c, anc, tuple(data)))
            anc += 1
    return checks


# ---------------------------------------------------------------------------
# Generic CSS container (used by the SAT synthesizer and generic scorer)
# ---------------------------------------------------------------------------


class CSS:
    def __init__(self, name, n_data, checks, logical_x, logical_z):
        self.name = name
        self.n = n_data
        # checks: list of (kind, tuple_of_data_qubits); ancilla = n + index
        self.checks = [(k, tuple(sorted(s)), n_data + i)
                       for i, (k, s) in enumerate(checks)]
        # logicals: one support, or a list of supports (k > 1 codes)
        as_list = (lambda L: [tuple(x) for x in L]
                   if L and not isinstance(L[0], int) else [tuple(L)])
        self.logicals_x = as_list(list(logical_x))
        self.logicals_z = as_list(list(logical_z))

    @property
    def logical_x(self):
        return self.logicals_x[0]

    @property
    def logical_z(self):
        return self.logicals_z[0]

    @property
    def edges(self):
        return [(i, q) for i, (_, sup, _) in enumerate(self.checks)
                for q in sup]


def css_from_matrices(name, hx_rows, hz_rows, n):
    """Build a CSS code from parity-check supports.

    hx_rows / hz_rows: list of X-/Z-check supports (iterables of qubit ids).
    Verifies CSS commutation, computes k = n - rank(Hx) - rank(Hz), and
    derives logical operator representatives by F2 linear algebra:
      logical Z reps = ker(Hx) / rowspace(Hz),
      logical X reps = ker(Hz) / rowspace(Hx).
    Representatives are arbitrary (not minimum weight); any representative
    is a valid observable for distance scoring."""
    from f2 import nullspace, quotient_basis, rank

    def mask(sup):
        m = 0
        for q in sup:
            m |= 1 << q
        return m

    hx = [mask(s) for s in hx_rows]
    hz = [mask(s) for s in hz_rows]
    for i, a in enumerate(hx):
        for j, b in enumerate(hz):
            if bin(a & b).count("1") & 1:
                raise ValueError(f"X-check {i} anticommutes with Z-check {j}")
    k = n - rank(hx) - rank(hz)
    lz = quotient_basis(nullspace(hx, n), hz)
    lx = quotient_basis(nullspace(hz, n), hx)
    assert len(lz) == k and len(lx) == k, (len(lz), len(lx), k)

    def sup(m):
        return tuple(q for q in range(n) if (m >> q) & 1)

    checks = ([("X", sup(m)) for m in hx] + [("Z", sup(m)) for m in hz])
    code = CSS(name, n, checks, [sup(m) for m in lx], [sup(m) for m in lz])
    code.k = k
    return code


def bb_code(l, m, a_monomials, b_monomials, name=None):
    """Bivariate bicycle code (Bravyi et al. 2024 conventions).

    Data qubits: two blocks L/R over the Z_l x Z_m torus, n = 2*l*m.
    A monomial x^a y^b maps site (i,j) -> (i+a mod l, j+b mod m).
    Hx = [A | B], Hz = [B^T | A^T]. Example [[72,12,6]]:
        bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])."""
    lm = l * m

    def site(i, j):
        return (i % l) * m + (j % m)

    hx, hz = [], []
    edge_class = {}     # (check_idx, data_qubit) -> translation class
    for i in range(l):
        for j in range(m):
            row = []
            for k, (a, b) in enumerate(a_monomials):
                q = site(i + a, j + b)
                row.append(q)
                edge_class[(len(hx), q)] = ("X", "A", k)
            for k, (a, b) in enumerate(b_monomials):
                q = lm + site(i + a, j + b)
                row.append(q)
                edge_class[(len(hx), q)] = ("X", "B", k)
            hx.append(row)
    nx = len(hx)
    for i in range(l):
        for j in range(m):
            row = []
            for k, (a, b) in enumerate(b_monomials):
                q = site(i - a, j - b)
                row.append(q)
                edge_class[(nx + len(hz), q)] = ("Z", "B", k)
            for k, (a, b) in enumerate(a_monomials):
                q = lm + site(i - a, j - b)
                row.append(q)
                edge_class[(nx + len(hz), q)] = ("Z", "A", k)
            hz.append(row)
    code = css_from_matrices(name or f"BB(l={l},m={m})", hx, hz, 2 * lm)
    # translation classes for orbit-tied scheduling (checks are listed
    # X block first, matching css_from_matrices)
    code.edge_class = edge_class
    code.l, code.m = l, m
    # torus layout: L block on even sites, R block offset diagonally;
    # X-check (i,j) between them horizontally, Z-check vertically
    code.coords = {}
    for i in range(l):
        for j in range(m):
            code.coords[i * m + j] = (2 * i, 2 * j)              # L data
            code.coords[lm + i * m + j] = (2 * i + 1, 2 * j + 1)  # R data
            code.coords[2 * lm + i * m + j] = (2 * i + 1, 2 * j)  # X anc
            code.coords[3 * lm + i * m + j] = (2 * i, 2 * j + 1)  # Z anc
    return code


def split_translation_classes(code, parts):
    """Refine a BB code's translation classes by check-site residue
    (i mod parts, j mod parts): 12 classes become 12*parts^2. Interpolates
    between fully symmetric schedules (parts=1) and per-check freedom --
    a coarser symmetry, so UNSAT proofs cover a strictly larger family
    and SAT may reveal schedules the full tying excludes. In place."""
    nx = code.l * code.m
    code.edge_class = {
        (ci, q): cls + ((ci % nx) // code.m % parts,
                        (ci % nx) % code.m % parts)
        for (ci, q), cls in code.edge_class.items()}
    return code


def describe(code):
    """Human-readable summary of a CSS code."""
    from collections import Counter
    wx = Counter(len(s) for k, s, _ in code.checks if k == "X")
    wz = Counter(len(s) for k, s, _ in code.checks if k == "Z")
    deg = Counter(q for _, q in code.edges)
    k = getattr(code, "k", len(code.logicals_z))
    lines = [
        f"{code.name}: n={code.n} data qubits, k={k} logical qubits",
        f"  X-checks: {sum(wx.values())} (weights {dict(wx)})",
        f"  Z-checks: {sum(wz.values())} (weights {dict(wz)})",
        f"  data-qubit degree: max {max(deg.values())} "
        f"(depth lower bound for scheduling)",
        f"  logical rep weights: Z {sorted(len(s) for s in code.logicals_z)}",
        f"                       X {sorted(len(s) for s in code.logicals_x)}",
    ]
    return "\n".join(lines)


def surface_css(d):
    """CSS container + raw Check list (with corner geometry) for d x d."""
    raw = build_code(d)
    checks = [(ch.kind, tuple(q for _, q in ch.data)) for ch in raw]
    code = CSS(f"surface d={d}", d * d, checks,
               logical_x=[r * d for r in range(d)],       # left column
               logical_z=list(range(d)))                   # top row
    code.coords = {r * d + c: (2 * c + 1, 2 * r + 1)
                   for r in range(d) for c in range(d)}
    code.coords.update({ch.anc: (2 * ch.c, 2 * ch.r) for ch in raw})
    return code, raw


def steane_css():
    supports = [(0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6)]
    checks = [("X", s) for s in supports] + [("Z", s) for s in supports]
    code = CSS("Steane [[7,1,3]]", 7, checks,
               logical_x=[0, 1, 2], logical_z=[0, 1, 2])
    # triangle layout (corners 0/1/3, edge midpoints 2/4/5, center 6);
    # X-ancillas offset left of their plaquette center, Z-ancillas right
    code.coords = {0: (0, 0), 1: (4, 0), 2: (2, 0), 3: (2, 4),
                   4: (1, 2), 5: (3, 2), 6: (2, 1)}
    centers = [(1.25, 0.75), (2.75, 0.75), (2, 2.25)]
    for i, (cx, cy) in enumerate(centers):
        code.coords[7 + i] = (cx - 0.25, cy - 0.25)      # X anc
        code.coords[10 + i] = (cx + 0.25, cy + 0.25)     # Z anc
    return code


def low_weight_logical_reps(code, kind, tries=6000, seed=0, keep=24):
    """Randomized search for low-weight logical representatives of `kind`
    ("Z": elements of ker(Hx) outside rowspace(Hz); "X" mirrored).
    Population XOR-descent over the normalizer basis plus stabilizer
    rows (adding stabilizers preserves the class, may lower the weight).
    Returns supports sorted by weight -- REPRESENTATIVES FOUND BY
    RANDOMIZED SEARCH, not certified minimum weight."""
    import random

    from f2 import nullspace, reduce_mod, rref

    def mask(sup):
        m = 0
        for q in sup:
            m |= 1 << q
        return m

    same = [mask(s) for k, s, _ in code.checks if k == kind]
    other = [mask(s) for k, s, _ in code.checks if k != kind]
    stab_piv = rref(same)
    pool = nullspace(other, code.n) + same
    rng = random.Random(seed)
    found = {}
    for v in pool:
        if reduce_mod(v, stab_piv):
            found[v] = bin(v).count("1")
    for _ in range(tries):
        v = rng.choice(pool) ^ rng.choice(pool)
        if not v:
            continue
        w = bin(v).count("1")
        heavy = max(range(len(pool)),
                    key=lambda i: bin(pool[i]).count("1"))
        if w < bin(pool[heavy]).count("1"):
            pool[heavy] = v
        if reduce_mod(v, stab_piv) and v not in found:
            found[v] = w
    reps = sorted(found, key=found.get)[:keep]
    return [tuple(q for q in range(code.n) if (v >> q) & 1) for v in reps]


def determined_first_checks(code, init_map, undetermined=()):
    """Check indices whose FIRST-round measurement is deterministic under a
    product-state initialization: every support qubit is initialized in the
    check's own basis, and neither the support nor the ancilla touches an
    `undetermined` qubit (one whose state is not a fixed stabilizer state,
    e.g. an injected magic state or its stand-in Pauli frame).

    init_map: {data_qubit: "Z"/"X"} (missing entries default to "Z").
    This is the candidate set for round-1 detector declaration; callers
    validate it against stim's own determinism check at p=0."""
    und = set(undetermined)
    out = []
    for i, (kind, sup, anc) in enumerate(code.checks):
        if anc in und or any(q in und for q in sup):
            continue
        if all(init_map.get(q, "Z") == kind for q in sup):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def layer_gates(checks, sched_z, sched_x, layer):
    """CNOTs (control, target) executed in the given layer (0..3) under
    uniform corner-permutation schedules.

    Z-check: data -> ancilla (ancilla is CNOT target).
    X-check: ancilla -> data (ancilla is CNOT control).
    A check skips a layer if its scheduled corner is absent (boundary check).
    """
    gates = []
    for ch in checks:
        sched = sched_z if ch.kind == "Z" else sched_x
        q = ch.corner_map().get(sched[layer])
        if q is None:
            continue
        gates.append((q, ch.anc) if ch.kind == "Z" else (ch.anc, q))
    return gates


def schedule_valid(checks, sched_z, sched_x):
    """Each qubit participates in at most one gate per layer."""
    for layer in range(4):
        seen = set()
        for a, b in layer_gates(checks, sched_z, sched_x, layer):
            if a in seen or b in seen:
                return False
            seen.add(a)
            seen.add(b)
    return True


def uniform_assignment(raw_checks, sz, sx):
    """Convert uniform corner permutations into a per-check edge->layer
    assignment {(check_index, data_qubit): layer}."""
    a = {}
    for i, ch in enumerate(raw_checks):
        perm = sz if ch.kind == "Z" else sx
        for corner, q in ch.data:
            a[(i, q)] = perm.index(corner)
    return a
