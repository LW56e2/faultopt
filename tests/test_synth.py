"""SAT synthesizer: encoding agreement with brute force, hook clauses,
depth proofs, libraries, orbit tying."""

import itertools

import pytest

from codes import (CORNERS, bb_code, schedule_valid, steane_css, surface_css,
                   uniform_assignment)
from scoring import deterministic, score_gates
from synth import (add_surface_hook_clauses, find_min_depth, make_se_synth)

N = ("NW", "SW", "NE", "SE")
Z = ("NW", "NE", "SW", "SE")


def test_sat_agrees_with_brute_force_sample():
    """SAT feasibility == (parallelism AND deterministic detectors) on a
    sample of uniform schedules including all shape-pair combinations."""
    code, raw = surface_css(3)
    synth = make_se_synth(code, 4, "sched")
    perms = list(itertools.permutations(CORNERS))
    sample = [(sz, sx) for sz in perms[:6] for sx in perms[:6]]
    sample += [(N, Z), (N, N), (Z, Z), (Z, N)]
    for sz, sx in sample:
        a = uniform_assignment(raw, sz, sx)
        truth = (schedule_valid(raw, sz, sx)
                 and deterministic(code, a, 4))
        assert synth.check_assignment(a) == truth, (sz, sx)


def test_depth_lower_bound_is_proven():
    code, _ = surface_css(3)
    assert not make_se_synth(code, 3, "sched").check()   # UNSAT at T=3
    assert make_se_synth(code, 4, "sched").check()       # SAT at T=4


def test_hookfree_space_is_exactly_the_full_distance_set():
    code, _ = surface_css(3)
    synth = make_se_synth(code, 4, "sched")
    add_surface_hook_clauses(synth, code, 3)
    sols = synth.enumerate(100)     # exhausts
    assert len(sols) == 16
    for layers in sols[:4]:         # spot-check scores
        assert score_gates(code, layers, 3) == (3, 3)


def test_steane_min_depth():
    T, _ = find_min_depth(steane_css())
    assert T == 6


def test_clifford_library_finds_solutions():
    code, _ = surface_css(3)
    synth = make_se_synth(code, 4, "clifford", max_gates=24)
    assert synth.check()
    layers = synth.model_layers()
    assert sum(len(l) for l in layers) <= 24
    # candidate must actually measure the code
    dz, dx = score_gates(code, layers, 3)
    assert dz >= 2 and dx >= 2


@pytest.mark.slow
def test_validate_all_576():
    code, raw = surface_css(3)
    synth = make_se_synth(code, 4, "sched")
    for sz, sx in itertools.product(itertools.permutations(CORNERS),
                                    repeat=2):
        a = uniform_assignment(raw, sz, sx)
        truth = (schedule_valid(raw, sz, sx)
                 and deterministic(code, a, 4))
        assert synth.check_assignment(a) == truth


@pytest.mark.slow
def test_bb72_symmetric_depth_proof():
    code = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)],
                   [(0, 3), (1, 0), (2, 0)])
    assert not make_se_synth(code, 6, "sched", tie_classes=True).check()
    assert make_se_synth(code, 7, "sched", tie_classes=True).check()


@pytest.mark.slow
def test_bb72_split2_depth6_unsat():
    """Depth-6 impossibility extends beyond full translation symmetry:
    UNSAT for the 48-class family (schedules invariant under all even
    translations), a strictly larger space than the 12-class proof."""
    from codes import split_translation_classes
    code = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)],
                   [(0, 3), (1, 0), (2, 0)])
    split_translation_classes(code, 2)
    assert not make_se_synth(code, 6, "sched", tie_classes=True).check()
