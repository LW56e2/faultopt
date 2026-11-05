"""CEGIS loop: propagation-tracking drift guard, witness extraction,
clause soundness audited against stim, end-to-end reproduction of the
surface-code ground truth."""

import pytest

from cegis import (assignment_from_layers, cegis_search, enumerate_matching,
                   propagate_tracking, witness_cone, witness_indices)
from codes import surface_css, steane_css, uniform_assignment
from scoring import layers_from_assignment, score_gates
from spectrum import (closed_signatures, declared_parities,
                      fragment_from_layers, min_weight)
from synth import add_surface_hook_clauses, make_se_synth

N = ("NW", "SW", "NE", "SE")
Z = ("NW", "NE", "SW", "SE")


def _uniform_layers(code, raw, sz, sx):
    return layers_from_assignment(code, uniform_assignment(raw, sz, sx), 4)


def test_assignment_from_layers_roundtrip():
    code, raw = surface_css(3)
    assign = uniform_assignment(raw, N, Z)
    layers = layers_from_assignment(code, assign, 4)
    assert assignment_from_layers(code, layers) == assign


def test_propagate_tracking_agrees_with_fragment_engine():
    """(flips, residual) from the tracking reimplementation must equal
    Fragment.propagate for EVERY mechanism, and the involved set must
    contain the mechanism's own support and its residual's support."""
    code, raw = surface_css(3)
    layers = _uniform_layers(code, raw, Z, Z)          # a bad schedule
    fr = fragment_from_layers(code, layers, 2, "Z", "Z")
    for i, (op_idx, paulis, _) in enumerate(fr.mechs):
        flips, residual = fr.propagate(op_idx, paulis)
        tflips, tres, involved = propagate_tracking(fr, i)
        assert (tflips, tres) == (flips, residual)
        assert {q for q, p in paulis.items() if p != (0, 0)} <= involved
        assert set(residual) <= involved


def _witness_ok(sigs, n_det, idxs, w):
    assert idxs is not None and len(idxs) == len(set(idxs)) == w
    x = 0
    for i in idxs:
        x ^= sigs[i]
    assert not x & ((1 << n_det) - 1) and x >> n_det


def test_witness_extraction_weights_2_and_3():
    """Real fragments: Z/Z has a weight-2 witness in the X basis (X-check
    hooks aligned), the N/Z winner a weight-3 one in Z; extracted indices
    must XOR to (det 0, obs != 0)."""
    code, raw = surface_css(3)
    for sched, basis, w in (((Z, Z), "X", 2), ((N, Z), "Z", 3)):
        det, obs = declared_parities(code, 3, basis)
        layers = _uniform_layers(code, raw, *sched)
        fr = fragment_from_layers(code, layers, 3, basis, basis)
        sigs, _ = closed_signatures(fr, det, obs)
        assert min_weight(sigs, len(det), 4) == w
        _witness_ok(sigs, len(det), witness_indices(sigs, len(det), w), w)


def test_witness_extraction_weight_4_synthetic():
    """d0, d1, d2, d0^d1^d2^obs: minimal witness is all four."""
    n_det = 3
    sigs = [0b001, 0b010, 0b100, 0b111 | (1 << n_det)]
    assert min_weight(sigs, n_det, 5) == 4
    _witness_ok(sigs, n_det, witness_indices(sigs, n_det, 4), 4)


def test_witness_extraction_weight_5_synthetic():
    """d0..d3, d0^d1^d2^d3^obs: minimal witness is all five."""
    n_det = 4
    sigs = [0b0001, 0b0010, 0b0100, 0b1000, 0b11111]
    assert min_weight(sigs, n_det, 6) == 5
    _witness_ok(sigs, n_det, witness_indices(sigs, n_det, 5), 5)


def test_witness_cone_is_valid_edge_pattern():
    code, raw = surface_css(3)
    layers = _uniform_layers(code, raw, Z, Z)
    assign = assignment_from_layers(code, layers)
    det, obs = declared_parities(code, 3, "X")
    fr = fragment_from_layers(code, layers, 3, "X", "X")
    sigs, _ = closed_signatures(fr, det, obs)
    idxs = witness_indices(sigs, len(det), 2)
    cone = witness_cone(code, fr, idxs)
    assert cone and set(cone) <= set(assign)
    assert len(cone) < len(assign)      # generalizes beyond one candidate


def test_cegis_finds_surface_winner_and_stim_agrees():
    code, _ = surface_css(3)
    res = cegis_search(code, 4, 3)
    assert res["status"] == "found" and len(res["winners"]) == 1
    assert score_gates(code, res["winners"][0]["layers"], 3) == (3, 3)
    # every non-winner iteration learned at least one clause
    assert res["learned_clauses"] >= res["iterations"] - 1


def test_learned_clauses_are_sound():
    """Every schedule matched by a learned pattern must really score below
    the target -- audited with the independent stim grader."""
    code, _ = surface_css(3)
    res = cegis_search(code, 4, 3, max_iters=3)
    assert res["patterns"]
    for pat in res["patterns"][:3]:
        matches = enumerate_matching(code, 4, pat["pattern"], 3)
        assert matches            # at least the candidate that spawned it
        for layers in matches:
            dz, dx = score_gates(code, layers, 3)
            assert min(dz, dx) < 3, (pat["witness"], dz, dx)


# Discovered BY the CEGIS loop (312 iterations; the earlier 300-sample
# enumeration missed it -- "sampled != exhaustive" made concrete): a
# depth-6 bare-ancilla Steane SE schedule with FULL circuit distance 3.
# Ancillas 7,8,9 = X-checks, 10,11,12 = Z-checks; 24 CNOTs = one per edge.
STEANE_T6_D3 = [
    [("CX", (8, 1)), ("CX", (9, 6)), ("CX", (0, 10)), ("CX", (2, 11)),
     ("CX", (3, 12))],
    [("CX", (7, 2)), ("CX", (9, 3)), ("CX", (1, 11)), ("CX", (6, 12))],
    [("CX", (7, 0)), ("CX", (8, 2)), ("CX", (9, 4)), ("CX", (6, 11)),
     ("CX", (5, 12))],
    [("CX", (7, 6)), ("CX", (8, 5)), ("CX", (2, 10)), ("CX", (4, 12))],
    [("CX", (8, 6)), ("CX", (9, 5)), ("CX", (4, 10))],
    [("CX", (7, 4)), ("CX", (6, 10)), ("CX", (5, 11))],
]


def test_steane_bare_ancilla_full_distance_schedule():
    """The CEGIS-discovered schedule really has d_circ = 3 in both bases
    (stim, exact) -- bare ancillas, minimum depth, no flags."""
    assert score_gates(steane_css(), STEANE_T6_D3, 3) == (3, 3)


@pytest.mark.slow
def test_cegis_rediscovers_steane_full_distance():
    """The loop finds SOME bare-ancilla depth-6 full-distance schedule
    from scratch (not necessarily the transcribed one)."""
    code = steane_css()
    res = cegis_search(code, 6, 3)
    assert res["status"] == "found"
    assert score_gates(code, res["winners"][0]["layers"], 3) == (3, 3)


@pytest.mark.slow
def test_cegis_exhaust_reproduces_the_16_winners():
    """Exhaust mode terminates (proof) with EXACTLY the 16 full-distance
    schedules -- the same set the web-derived hook clauses select."""
    code, _ = surface_css(3)
    res = cegis_search(code, 4, 3, exhaust=True)
    assert res["status"] == "exhausted" and res["proof"]
    found = {frozenset(assignment_from_layers(code, w["layers"]).items())
             for w in res["winners"]}
    assert len(found) == 16
    synth = make_se_synth(code, 4, "sched")
    add_surface_hook_clauses(synth, code, 3)
    truth = {frozenset(assignment_from_layers(code, layers).items())
             for layers in synth.enumerate(100)}
    assert found == truth
