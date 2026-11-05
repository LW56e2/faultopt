"""Packed min-weight kernel: exact equivalence with spectrum.min_weight."""

from pathlib import Path

import json
import random

import pytest

from codes import bb_code, steane_css, surface_css, uniform_assignment
from mim import min_weight_packed
from scoring import layers_from_assignment
from spectrum import (closed_signatures, declared_parities,
                      fragment_from_layers, min_weight)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _sigs(code, layers, rounds, basis):
    fr = fragment_from_layers(code, layers, rounds, basis, basis)
    det, obs = declared_parities(code, rounds, basis)
    s, _ = closed_signatures(fr, det, obs)
    return s, len(det)


def test_randomized_equivalence():
    random.seed(1)
    for _ in range(150):
        n_det = random.randint(3, 24)
        n_obs = random.randint(1, 3)
        sigs = [random.getrandbits(n_det + n_obs)
                for _ in range(random.randint(4, 100))]
        for cutoff in (2, 3, 4, 5, 6):
            assert (min_weight(sigs, n_det, cutoff)
                    == min_weight_packed(sigs, n_det, cutoff))


def test_real_cases_match():
    code, raw = surface_css(3)
    for sz, sx in ((("NW", "SW", "NE", "SE"), ("NW", "NE", "SW", "SE")),
                   (("NW", "NE", "SW", "SE"), ("NW", "SW", "NE", "SE"))):
        L = layers_from_assignment(code, uniform_assignment(raw, sz, sx), 4)
        for basis in "ZX":
            s, nd = _sigs(code, L, 3, basis)
            for cutoff in (4, 5, 6):
                assert (min_weight(s, nd, cutoff)
                        == min_weight_packed(s, nd, cutoff))
    st = steane_css()
    with open(ARTIFACTS / "steane_t6_winner.json") as f:
        L = [[(n, tuple(q)) for n, q in layer]
             for layer in json.load(f)["circuits"][0]]
    for basis in "ZX":
        s, nd = _sigs(st, L, 3, basis)
        assert min_weight(s, nd, 6) == min_weight_packed(s, nd, 6) == 3


@pytest.mark.slow
def test_bb72_scale_cutoff5_matches():
    code = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)],
                   [(0, 3), (1, 0), (2, 0)])
    with open(ARTIFACTS / "bb72_t7_tied_d4.json") as f:
        L = [[(n, tuple(q)) for n, q in layer]
             for layer in json.load(f)["circuits"][0]]
    s, nd = _sigs(code, L, 2, "Z")
    assert min_weight(s, nd, 5) == min_weight_packed(s, nd, 5) == 4
