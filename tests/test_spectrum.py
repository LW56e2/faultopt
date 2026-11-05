"""Open scorer: backend agreement with stim, the cut test, spectra."""

from pathlib import Path

import pytest

from codes import surface_css, uniform_assignment
from grader import get_grader
from scoring import layers_from_assignment
from spectrum import (WebGrader, compose_cut, declared_parities, dominates,
                      fragment_from_layers, min_weight,
                      restrict_undetectable, spectrum_table,
                      window_signatures)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

N = ("NW", "SW", "NE", "SE")
Z = ("NW", "NE", "SW", "SE")


def layers_for(sz, sx):
    code, raw = surface_css(3)
    return code, layers_from_assignment(
        code, uniform_assignment(raw, sz, sx), 4)


# -- unit level -------------------------------------------------------------


def test_min_weight_crafted():
    # 2 detector bits, 1 obs bit (bit 2)
    assert min_weight([0b100], 2) == 1                    # obs, no dets
    assert min_weight([0b101, 0b001], 2) == 2             # pair cancels det
    # triple: 101 ^ 010 ^ 011 = 100 (obs set, dets clear); no single/pair
    assert min_weight([0b101, 0b010, 0b011], 2) == 3
    assert min_weight([0b001, 0b010], 2, cutoff=4) == 4   # nothing -> >=4


def test_min_weight_five_and_cutoff_six():
    """d0,d1,d2,d3,d0^d1^d2^d3^obs: only the full 5-set is undetectable
    and logical. Cutoff 5 certifies ">= 5"; cutoff 6 resolves it exactly."""
    import pytest as _pytest
    sigs = [0b0001, 0b0010, 0b0100, 0b1000, 0b11111]      # n_det=4, obs=bit4
    assert min_weight(sigs, 4, cutoff=5) == 5             # ">= 5"
    assert min_weight(sigs, 4, cutoff=6) == 5             # exact
    assert min_weight([0b0001, 0b0010], 4, cutoff=6) == 6  # nothing -> >=6
    with _pytest.raises(ValueError):
        min_weight(sigs, 4, cutoff=7)


def test_dominates_crafted():
    A = {0: 0, 5: 2}
    B = {0: 0, 5: 1}
    ok, _ = dominates(A, B)          # A prices >= B prices: A safer
    assert ok
    ok, ex = dominates(B, A)
    assert not ok and ex == 5


# -- backend agreement -------------------------------------------------------


def test_webs_grader_matches_stim():
    code, raw = surface_css(3)
    stim_g, webs_g = get_grader("stim"), get_grader("webs")
    for sz, sx in [(N, Z), (N, N), (Z, N), (Z, Z)]:
        layers = layers_from_assignment(
            code, uniform_assignment(raw, sz, sx), 4)
        assert stim_g.score(code, layers, 3) == \
            webs_g.score(code, layers, 3)


def test_webs_measures_code_agrees_with_stim():
    import itertools
    from codes import CORNERS, schedule_valid
    code, raw = surface_css(3)
    stim_g, webs_g = get_grader("stim"), get_grader("webs")
    checked = 0
    for sz, sx in itertools.product(itertools.permutations(CORNERS),
                                    repeat=2):
        if not schedule_valid(raw, sz, sx):
            continue
        layers = layers_from_assignment(
            code, uniform_assignment(raw, sz, sx), 4)
        assert (webs_g.measures_code(code, layers)
                == stim_g.measures_code(code, layers))
        checked += 1
        if checked >= 20:
            break


def test_webs_explain_finds_hook_witness():
    code, layers = layers_for(N, N)
    basis, witness = get_grader("webs").explain(code, layers, 3)
    assert basis == "Z" and len(witness) == 2


# -- the cut test -------------------------------------------------------------


def test_cut_test_reproduces_stim():
    code, raw = surface_css(3)
    det, obs = declared_parities(code, 3, "Z")
    stim_g = get_grader("stim")
    for sz, sx, expect in [(N, Z, 3), (N, N, 2)]:
        layers = layers_from_assignment(
            code, uniform_assignment(raw, sz, sx), 4)
        fr1 = fragment_from_layers(code, layers, 2, "Z", None)
        fr2 = fragment_from_layers(code, layers, 1, None, "Z",
                                   anc_init_noise=False)
        w, _, _ = compose_cut(code, fr1, fr2, 16, det, obs)
        assert w == expect
        assert w == stim_g.score(code, layers, 3)[0]


def test_window_spectra_incomparable_without_context():
    code, raw = surface_css(3)
    det, _ = declared_parities(code, 3, "Z")
    local = [m & 0xFFFF for m in det if m & 0xFFFF]
    n_internal = sum(1 for m in det if m and not (m >> 16))
    mask = (1 << n_internal) - 1
    tables = {}
    for sz, sx in [(N, Z), (N, N)]:
        layers = layers_from_assignment(
            code, uniform_assignment(raw, sz, sx), 4)
        fr1 = fragment_from_layers(code, layers, 2, "Z", None)
        wsigs, _, _ = window_signatures(fr1, local, list(range(9)))
        tables[(sz, sx)] = restrict_undetectable(
            spectrum_table([s for s, _ in wsigs]), mask)
    fwd, _ = dominates(tables[(N, Z)], tables[(N, N)])
    rev, _ = dominates(tables[(N, N)], tables[(N, Z)])
    assert not fwd and not rev


@pytest.mark.slow
def test_webs_agrees_with_stim_on_bb72():
    """First qLDPC-scale cross-validation, on the stored tied depth-7
    BB72 candidate (bb72_t7_tied_d4.json): webs (cutoff 5) and stim both
    give distance 4 per basis on the 2-round window. (A fresh z3 model is
    NOT used here: stim's search cost explodes on higher-distance
    candidates, making the runtime unpredictable.)"""
    import json
    from codes import bb_code
    from scoring import circuit_distance, circuit_from_gates
    from spectrum import closed_signatures, fragment_from_layers, min_weight
    code = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)],
                   [(0, 3), (1, 0), (2, 0)])
    path = ARTIFACTS / "bb72_t7_tied_d4.json"
    with open(path) as f:
        layers = [[(n, tuple(q)) for n, q in layer]
                  for layer in json.load(f)["circuits"][0]]
    for basis in "ZX":
        det, obs = declared_parities(code, 2, basis)
        fr = fragment_from_layers(code, layers, 2, basis, basis)
        sigs, _ = closed_signatures(fr, det, obs)
        w = min_weight(sigs, len(det), 5)
        ref, _ = circuit_distance(circuit_from_gates(code, layers, basis, 2))
        assert w == ref == 4, (basis, w, ref)
