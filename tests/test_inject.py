"""Injection census: pinned ground truth for standard and hook magic-state
injection at d=3, declaration-derivation correctness, webs-vs-stim
agreement, and the frame-reduction identities."""

import json

import pytest

from codes import (bb_code, determined_first_checks, low_weight_logical_reps,
                   surface_css, uniform_assignment)
from inject import (GOOD_SCHED, anticommutes, hook_frames, injection_report,
                    logical_pauli, standard_frames, surface_pattern,
                    zcheck_hook_config, zcheck_hook_frames)
from scoring import layers_from_assignment

from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

BAD_SCHED = (("NW", "NE", "SW", "SE"), ("NW", "SW", "NE", "SE"))
WIN1 = (("NE", "SE", "NW", "SW"), ("NE", "NW", "SE", "SW"))
MONO_A = [(3, 0), (0, 1), (0, 2)]
MONO_B = [(0, 3), (1, 0), (2, 0)]


@pytest.fixture(scope="module")
def surface3():
    code, raw = surface_css(3)
    layers = layers_from_assignment(
        code, uniform_assignment(raw, *GOOD_SCHED), 4)
    return code, raw, layers


@pytest.fixture(scope="module")
def standard_corner(surface3):
    code, _, layers = surface3
    frames = standard_frames(code, layers, 0,
                             surface_pattern(3, 0, 0, "diag"), 2)
    report, results, kept = injection_report(code, frames)
    return code, frames, report, results, kept


@pytest.fixture(scope="module")
def hook_corner(surface3):
    code, _, layers = surface3
    frames, pair = hook_frames(code, 3, layers, 2)
    report, results, kept = injection_report(code, frames)
    return code, frames, pair, report, results, kept


def test_standard_corner_census(standard_corner):
    _, _, report, _, _ = standard_corner
    by = {f["frame"]: f for f in report["frames"]}
    assert by["frameZ"]["w1_count"] == 11
    assert by["frameX"]["w1_count"] == 17
    assert report["union"]["w1_count"] == 24
    assert report["union"]["w1_weighted"] == pytest.approx(4.8)
    assert all(f["stim_agrees"] for f in report["frames"])


def test_standard_corner_declarations(standard_corner):
    code, frames, report, _, kept = standard_corner
    # round-1 declared singles == the init-determined checks with the
    # injection site treated as undetermined (codes.determined_first_checks)
    ones0 = sorted(e[2] for e in kept if e[0] == "one" and e[1] == 0)
    pred = determined_first_checks(code, frames[0]["init_map"],
                                   undetermined={0})
    assert ones0 == pred == [3, 6, 7]
    # every dropped candidate is a single-record parity; compares and the
    # noiseless final readout all survive
    assert all(e[0] == "one" for e in report["detectors_dropped"])
    assert sum(1 for e in kept if e[0] == "cmp") == 16
    assert sum(1 for e in kept if e[0] == "fin") == 8


def test_standard_observables_need_no_folding(standard_corner):
    _, _, report, _, _ = standard_corner
    # the corner pattern anchors both logical strings directly
    assert [f["folded_records"] for f in report["frames"]] == [0, 0]


def test_hook_corner_census(hook_corner):
    _, _, pair, report, _, _ = hook_corner
    assert pair == (0, 3)
    by = {f["frame"]: f for f in report["frames"]}
    assert by["frameZ"]["w1_count"] == 4
    assert by["frameY"]["w1_count"] == 7
    assert report["union"]["w1_count"] == 7
    assert report["union"]["w1_weighted"] == pytest.approx(19 / 15)
    # the teleportation correction is exactly one folded ancilla record
    assert by["frameY"]["folded_records"] == 1
    assert all(f["stim_agrees"] for f in report["frames"])


def test_hook_corrupting_mechanisms_touch_injection_only(hook_corner):
    _, _, _, report, _, _ = hook_corner
    # every accepted corrupting single lives in round 0 on the hook
    # ancilla's gates (the unprotected injection moment)
    assert all(" r0 " in lab or lab.endswith("r0 X") or "r0" in lab
               for lab in report["union"]["w1_labels"])


def test_hook_beats_standard(standard_corner, hook_corner):
    _, _, std, _, _ = standard_corner
    _, _, _, hook, _, _ = hook_corner
    assert hook["union"]["w1_count"] < std["union"]["w1_count"]
    assert hook["union"]["w1_weighted"] < std["union"]["w1_weighted"]


def test_growth_schedule_moves_weight2(surface3):
    code, raw, good = surface3
    win1 = layers_from_assignment(
        code, uniform_assignment(raw, *WIN1), 4)
    bad = layers_from_assignment(
        code, uniform_assignment(raw, *BAD_SCHED), 4)
    reports = {}
    for name, L in (("good", good), ("win1", win1), ("bad", bad)):
        frames, _ = hook_frames(code, 3, L, 2)
        reports[name], _, _ = injection_report(code, frames,
                                               stim_check=False)
    by = {name: {f["frame"]: f for f in rep["frames"]}
          for name, rep in reports.items()}
    # weight-1 is injection-window-dominated (identical), weight-2 prices
    # the growth schedule -- the joint-optimization signal.  The four
    # memory-optimal schedules are NOT injection-equivalent: win1
    # (NE,SE,NW,SW) beats win0 at frameZ weight 2, and the memory-bad
    # zigzag is strictly worst.
    assert (by["good"]["frameZ"]["w1_count"]
            == by["win1"]["frameZ"]["w1_count"]
            == by["bad"]["frameZ"]["w1_count"])
    assert (by["win1"]["frameZ"]["w2_pairs"]
            < by["good"]["frameZ"]["w2_pairs"]
            < by["bad"]["frameZ"]["w2_pairs"])
    assert by["good"]["frameY"]["w2_pairs"] < by["bad"]["frameY"]["w2_pairs"]
    assert by["win1"]["frameZ"]["w2_pairs"] == 810


def test_hook_only_subpatch_logical_injects(surface3):
    """Position/orientation sweep: every tail that is not a full column
    of the 2x2 block (= the subpatch logical) is mechanically rejected."""
    code, _, layers = surface3
    for tail in ((0, 1), (3, 4), (0,), (3,), (1, 3, 4)):
        frames, _ = hook_frames(code, 3, layers, 2, tail=tail)
        with pytest.raises(ValueError, match="does not inject"):
            injection_report(code, frames, stim_check=False)


def test_d5_hook_census_distance_independent():
    """The weight-1 hook census is a property of the injection gadget,
    not the code distance: d=5 reproduces d=3's 7 locations at 19p/15."""
    code, raw = surface_css(5)
    layers = layers_from_assignment(
        code, uniform_assignment(raw, *WIN1), 4)
    frames, pair = hook_frames(code, 5, layers, 2)
    assert pair == (0, 5)
    report, _, _ = injection_report(code, frames)
    by = {f["frame"]: f for f in report["frames"]}
    assert by["frameZ"]["w1_count"] == 4
    assert by["frameY"]["w1_count"] == 7
    assert report["union"]["w1_count"] == 7
    assert report["union"]["w1_weighted"] == pytest.approx(19 / 15)
    assert all(f["stim_agrees"] for f in report["frames"])


def test_lao_criger_reproduction(surface3):
    """CF'22 (DOI 10.1145/3528416.3530237) Eqs (2)-(3) under their model
    and |Y>-state criterion, on their exact init patterns (mirrored onto
    this repo's lattice coloring): the structure reproduces exactly --
    2 p_IN for corner injection on precisely the two qubits they name
    (magic qubit + its neighbor), 1 p_IN for middle, zero measurement
    terms -- and the Tomita-Svore winner ordering gives two-qubit
    coefficients 8/15 (CR) and 7/15 (MR), IMPROVING their published
    9/15 for both (their custom sensitive-qubit orderings differ)."""
    from inject import lc_census
    code, _, layers = surface3
    cr = lc_census(code, layers, 2,
                   {0: "Z", 1: "Z", 3: "Z", 4: "X", 5: "X", 6: "X",
                    7: "X", 8: "X"}, stim_check=True)
    assert (cr["p2_fifteenths"], cr["p_in"], cr["p_m"],
            cr["p1_thirds"]) == (8, 2, 0, 0)
    assert "init flip q2" in cr["corrupting_mechanisms"]   # magic qubit
    assert "init flip q1" in cr["corrupting_mechanisms"]   # their D6
    assert cr["frame_detect_mismatches"] == 0
    assert cr["stim_agrees"]
    mr = lc_census(code, layers, 4,
                   {0: "Z", 1: "X", 2: "X", 3: "Z", 5: "Z", 6: "X",
                    7: "X", 8: "Z"}, stim_check=True)
    assert (mr["p2_fifteenths"], mr["p_in"], mr["p_m"],
            mr["p1_thirds"]) == (7, 1, 0, 0)
    assert mr["corrupting_mechanisms"].count("init flip q4") == 1
    assert mr["frame_detect_mismatches"] == 0
    assert mr["stim_agrees"]


def test_lao_criger_optimum_artifacts(surface3):
    """The committed optimum schedules from the exhaustive 2704-schedule
    sweep (the archived injection sweep): CR = 8/15 p2 (a value below Lao-Criger's
    published 9/15, which NO valid depth-4 schedule realizes), MR = 7/15
    (vs their realizable-but-suboptimal 9/15)."""
    import json

    from inject import lc_census
    code, _, _ = surface3
    for fname, site, im, expect in (
            ("lc_cr_optimum.json", 2,
             {0: "Z", 1: "Z", 3: "Z", 4: "X", 5: "X", 6: "X", 7: "X",
              8: "X"}, (8, 2, 0, 0)),
            ("lc_mr_optimum.json", 4,
             {0: "Z", 1: "X", 2: "X", 3: "Z", 5: "Z", 6: "X", 7: "X",
              8: "Z"}, (7, 1, 0, 0))):
        with open(ARTIFACTS / fname) as f:
            layers = [[(n, tuple(q)) for n, q in layer]
                      for layer in json.load(f)["circuits"][0]]
        c = lc_census(code, layers, site, im, stim_check=True)
        assert (c["p2_fifteenths"], c["p_in"], c["p_m"],
                c["p1_thirds"]) == expect
        assert c["stim_agrees"]


def test_closed_census_is_the_distance_restated(surface3):
    """The census criterion applied to CLOSED memory experiments: a
    full-distance d=3 schedule has zero corrupting singles and pairs;
    the hook-limited zigzag (d_eff = 2) has zero singles but a nonzero
    pair count -- the census is the counting refinement of d_circ (the
    first weight where it is nonzero), and prices HOW bad a
    distance-deficient schedule is (the p^2 coefficient)."""
    from inject import closed_census
    code, raw, good = surface3
    bad = layers_from_assignment(
        code, uniform_assignment(raw, *BAD_SCHED), 4)
    for basis in ("Z", "X"):
        c = closed_census(code, good, 3, basis)
        assert c["w1_count"] == 0 and c["w2_pairs"] == 0
    cz = closed_census(code, bad, 3, "Z")
    cx = closed_census(code, bad, 3, "X")
    assert cz["w1_count"] == cx["w1_count"] == 0
    assert (cz["w2_pairs"], cx["w2_pairs"]) == (262, 330)
    assert cz["w2_weighted"] == pytest.approx(2.8)


def test_zcheck_hook_config_algebra():
    """The generic hook configuration satisfies its defining F2
    conditions on BB [[72,12,6]]."""
    code = bb_code(6, 6, MONO_A, MONO_B)
    reps = low_weight_logical_reps(code, "Z")
    assert all(len(r) == 6 for r in reps[:12])      # d = 6 reps found
    lz = reps[0]
    ci, tail, lx = zcheck_hook_config(code, lz)
    kind, sup, _ = code.checks[ci]
    assert kind == "Z"
    assert set(tail) <= set(sup) & set(lz)
    R = set(lz) - set(tail)
    assert not (set(lx) & R)
    assert len(set(lx) & set(tail)) % 2 == 1
    assert len(set(lx) & set(sup)) % 2 == 0
    # lx commutes with every Z-check (it is an X-normalizer element)
    for k, s, _ in code.checks:
        if k == "Z":
            assert len(set(lx) & set(s)) % 2 == 0
    # lz commutes with every X-check
    for k, s, _ in code.checks:
        if k == "X":
            assert len(set(lz) & set(s)) % 2 == 0


@pytest.mark.slow
def test_bb72_hook_census():
    """First qLDPC hook-injection census, pinned: minimal round 0 leaves
    the spread-out logical exposed (69 locations); measuring the safe
    X-checks in the injection round cuts it to 40 (12p), stim-confirmed
    location by location."""
    code = bb_code(6, 6, MONO_A, MONO_B, name="BB [[72,12,6]]")
    path = ARTIFACTS / "bb72_cegis_winners.json"
    with open(path) as f:
        growth = [[(n, tuple(q)) for n, q in layer]
                  for layer in json.load(f)["circuits"][-1]]
    lz = low_weight_logical_reps(code, "Z")[0]
    ci, tail, lx = zcheck_hook_config(code, lz)
    assert (ci, tail) == (37, (36,))

    frames = zcheck_hook_frames(code, ci, tail, lz, lx, growth, 2,
                                r0_checks="inj")
    rep, _, _ = injection_report(code, frames, stim_check=False)
    assert rep["union"]["w1_count"] == 69
    assert rep["union"]["w1_weighted"] == pytest.approx(16.333333)

    frames = zcheck_hook_frames(code, ci, tail, lz, lx, growth, 2,
                                r0_checks="xsafe")
    rep, _, _ = injection_report(code, frames, stim_check=True)
    by = {f["frame"]: f for f in rep["frames"]}
    assert rep["union"]["w1_count"] == 40
    assert rep["union"]["w1_weighted"] == pytest.approx(12.0)
    assert by["frameY"]["folded_records"] == 1
    assert all(f["stim_agrees"] for f in rep["frames"])

    # the same criterion on the SE schedule itself: full distance 6
    # (R10), so the memory census is empty at weights 1 and 2
    from inject import closed_census
    for basis in ("Z", "X"):
        c = closed_census(code, growth, 2, basis)
        assert c["mechanisms"] == 13752
        assert c["w1_count"] == 0 and c["w2_pairs"] == 0


def test_y_coverage_identity(standard_corner):
    """A residual anticommutes with Y_L iff it anticommutes with exactly
    one of X_L, Z_L -- the reduction that lets two Pauli frames cover the
    whole logical action (docs/REPRODUCIBILITY.md)."""
    code, frames, _, _, _ = standard_corner
    xl, zl, yl = (logical_pauli(code, k) for k in "XZY")
    fr = frames[0]["fragment"]
    seen = 0
    for op_idx, paulis, _ in fr.mechs[:200]:
        _, residual = fr.propagate(op_idx, paulis)
        if residual:
            seen += 1
            assert anticommutes(residual, yl) == (
                anticommutes(residual, xl) ^ anticommutes(residual, zl))
    assert seen > 50


def test_uninjectable_pattern_rejected(surface3):
    code, _, layers = surface3
    frames = standard_frames(code, layers, 0,
                             {q: "Z" for q in range(1, code.n)}, 2)
    with pytest.raises(ValueError, match="does not inject"):
        injection_report(code, frames)


def test_hook_needs_x_type_subpatch(surface3):
    code, _, layers = surface3
    with pytest.raises(ValueError, match="X-check"):
        hook_frames(code, 3, layers, 2, anchor=1)
