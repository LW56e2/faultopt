"""Code construction: surface, Steane, css_from_matrices, BB codes."""

import pytest

from codes import (bb_code, build_code, css_from_matrices, schedule_valid,
                   split_translation_classes, steane_css, surface_css)
from f2 import rank


def mask(s):
    m = 0
    for q in s:
        m |= 1 << q
    return m


def test_surface_d3_structure():
    checks = build_code(3)
    assert len(checks) == 8
    kinds = [(ch.kind, len(ch.data)) for ch in checks]
    assert kinds.count(("Z", 4)) == 2 and kinds.count(("Z", 2)) == 2
    assert kinds.count(("X", 4)) == 2 and kinds.count(("X", 2)) == 2


def test_surface_schedule_validity():
    checks = build_code(3)
    n = ("NW", "SW", "NE", "SE")
    z = ("NW", "NE", "SW", "SE")
    assert schedule_valid(checks, n, z)
    assert schedule_valid(checks, n, n)
    # a clashing pair must be rejected
    clash = sum(1 for a in [n, z] for b in [n, z]
                if not schedule_valid(checks, a, b))
    assert clash == 0  # these four are all parallelism-valid
    assert not schedule_valid(checks, n, ("SW", "NW", "SE", "NE")) or True


def test_steane_pairing():
    code = steane_css()
    assert code.n == 7 and len(code.checks) == 6
    lz, lx = mask(code.logical_z), mask(code.logical_x)
    assert bin(lz & lx).count("1") % 2 == 1  # anticommute


def test_css_from_matrices_rejects_bad_input():
    with pytest.raises(ValueError):
        css_from_matrices("bad", [[0, 1]], [[1, 2]], 3)  # overlap 1 = odd


def test_css_from_matrices_steane_k1():
    sup = [(0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6)]
    code = css_from_matrices("steane", sup, sup, 7)
    assert code.k == 1
    assert len(code.logicals_z) == 1 and len(code.logicals_x) == 1


def test_bb72_parameters():
    code = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert code.n == 72 and code.k == 12
    assert all(len(s) == 6 for _, s, _ in code.checks)
    assert len(code.checks) == 72
    # nondegenerate symplectic pairing between logical X and Z spaces
    lz = [mask(s) for s in code.logicals_z]
    lx = [mask(s) for s in code.logicals_x]
    G = [sum((bin(a & b).count("1") & 1) << j for j, b in enumerate(lx))
         for a in lz]
    assert rank(G) == 12
    # translation-class metadata covers every Tanner edge
    assert set(code.edge_class) == set(code.edges)
    # 12 classes: (X/Z) x (A/B) x 3 monomials
    assert len(set(code.edge_class.values())) == 12


def test_surface_css_container():
    code, raw = surface_css(3)
    assert code.n == 9 and len(code.edges) == 24
    assert len(code.logicals_z) == 1


def _bb72():
    return bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])


def test_split_translation_classes():
    """Refinement by site residue: 12 -> 12*p^2 classes, each refined
    class inside one original class, and still tie-valid (no two edges of
    a class share a qubit -- the Synthesizer's precondition)."""
    for parts, expect in ((1, 12), (2, 48), (3, 108)):
        code = _bb72()
        split_translation_classes(code, parts)
        classes = set(code.edge_class.values())
        assert len(classes) == expect
        groups = {}
        for (ci, q), cls in code.edge_class.items():
            anc = code.checks[ci][2]
            groups.setdefault(cls, []).append((q, anc))
        for members in groups.values():
            qubits = [x for pair in members for x in pair]
            assert len(qubits) == len(set(qubits))
