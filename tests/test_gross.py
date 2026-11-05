"""Gross code campaign artifacts: the depth-7 winner and the depth
proofs (docs/REPRODUCIBILITY.md). UNSAT re-verification is minutes-scale and lives
in the reproduce commands, not here; these tests pin the artifacts."""

from pathlib import Path

import json

import pytest

from codes import bb_code
from grader import get_grader
from mim import min_weight_packed
from spectrum import (closed_signatures, declared_parities,
                      fragment_from_layers)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

MONO_A = [(3, 0), (0, 1), (0, 2)]
MONO_B = [(0, 3), (1, 0), (2, 0)]


def gross():
    return bb_code(12, 6, MONO_A, MONO_B, name="gross BB(12,6)")


def winner_layers():
    with open(ARTIFACTS / "gross_cegis_t6_gpu.json") as f:
        return [[(n, tuple(q)) for n, q in layer]
                for layer in json.load(f)["circuits"][0]]


def test_gross_artifacts_consistent():
    code = gross()
    assert code.n == 144 and code.k == 12
    L = winner_layers()
    assert len(L) == 7
    assert sum(len(x) for x in L) == 864          # every Tanner edge once
    # depth-proof artifacts: T=6 UNSAT for tied and both split families
    for f in ("gross_T6_tied.json", "gross_T6_split2.json",
              "gross_T6_split3.json"):
        with open(ARTIFACTS / f) as fh:
            assert json.load(fh)["sat"] is False, f
    # the target-5 and target-6 runs certified the SAME schedule
    with open(ARTIFACTS / "gross_cegis_t5_run1.json") as f:
        assert json.load(f)["circuits"][0] == \
            json.load(open(ARTIFACTS / "gross_cegis_t6_gpu.json"))["circuits"][0]


def test_gross_winner_measures_code():
    assert get_grader("webs").measures_code(gross(), winner_layers())


@pytest.mark.slow
def test_gross_winner_certified_at_least_5():
    """Re-derive the certified >=5 (cutoff-5 packed kernel; the >=6 leg
    needs the GPU sweep and is pinned by artifact + reproduce command)."""
    code = gross()
    L = winner_layers()
    for basis in "ZX":
        fr = fragment_from_layers(code, L, 2, basis, basis)
        det, obs = declared_parities(code, 2, basis)
        sigs, _ = closed_signatures(fr, det, obs)
        assert min_weight_packed(sigs, len(det), 5, procs=4) == 5   # >= 5
