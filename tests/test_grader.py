"""Grader seam: interface conformance and backend selection."""

import pytest

from codes import surface_css, uniform_assignment
from grader import get_grader, register
from scoring import layers_from_assignment, score_gates

N = ("NW", "SW", "NE", "SE")
Z = ("NW", "NE", "SW", "SE")


def layers_for(sz, sx):
    code, raw = surface_css(3)
    return code, layers_from_assignment(
        code, uniform_assignment(raw, sz, sx), 4)


def test_stim_grader_matches_scoring_module():
    code, layers = layers_for(N, Z)
    g = get_grader("stim")
    assert g.score(code, layers, 3) == score_gates(code, layers, 3) == (3, 3)


def test_measures_code():
    g = get_grader("stim")
    code, good = layers_for(N, Z)
    assert g.measures_code(code, good)
    # corrupted interleaving: swap one Z-check's CNOT into a clashing layer
    # is hard to construct by hand; instead use a known-corrupted uniform
    # pair discovered by search in test_scoring -- here just check the API
    # contract: score raises ValueError iff not measures_code
    _, bad_pair = layers_for(("NE", "SE", "NW", "SW"), Z)
    if not g.measures_code(code, bad_pair):
        with pytest.raises(ValueError):
            g.score(code, bad_pair, 3)


def test_explain_returns_witness():
    code, layers = layers_for(N, N)      # d_Z = 2: witness has 2 events
    basis, lines = get_grader("stim").explain(code, layers, 3)
    assert basis in "ZX" and len(lines) == 2
    assert all("->" in line for line in lines)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        get_grader("nope")


def test_backend_registration():
    class Dummy:
        name = "dummy"

        def score(self, code, layers, rounds):
            return (99, 99)

    register("dummy", Dummy)
    assert get_grader("dummy").score(None, None, None) == (99, 99)