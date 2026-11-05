"""Closed grader: known distances of known schedules (the demo ground truth
lives here, as assertions, so demos can stay assertion-free)."""

import itertools

import pytest

from codes import (CORNERS, build_code, schedule_valid, surface_css,
                   uniform_assignment)
from scoring import (circuit_distance, deterministic, layers_from_assignment,
                     memory_circuit, score_assignment, score_gates,
                     score_schedule)

N = ("NW", "SW", "NE", "SE")   # ends on vertical pair
Z = ("NW", "NE", "SW", "SE")   # ends on horizontal pair


def test_good_schedule_full_distance():
    checks = build_code(3)
    dz, _ = circuit_distance(memory_circuit(3, checks, N, Z, "Z", 3))
    dx, _ = circuit_distance(memory_circuit(3, checks, N, Z, "X", 3))
    assert (dz, dx) == (3, 3)


def test_bad_schedule_hook_limited():
    checks = build_code(3)
    dz, _ = circuit_distance(memory_circuit(3, checks, N, N, "Z", 3))
    assert dz == 2


def test_score_schedule_flags_corrupted_interleaving():
    # find a parallelism-valid schedule with non-deterministic detectors;
    # score_schedule must report d_eff None for it
    code, raw = surface_css(3)
    checks = raw
    for sz, sx in itertools.product(itertools.permutations(CORNERS),
                                    repeat=2):
        if not schedule_valid(checks, sz, sx):
            continue
        if not deterministic(code, uniform_assignment(raw, sz, sx), 4):
            res = score_schedule((3, 3, sz, sx))
            assert res["d_eff"] is None
            return
    pytest.fail("no corrupted-interleaving schedule found")


def test_assignment_path_matches_uniform_path():
    code, raw = surface_css(3)
    a = uniform_assignment(raw, N, Z)
    assert score_assignment(code, a, 3, 4) == (3, 3)
    assert score_gates(code, layers_from_assignment(code, a, 4), 3) == (3, 3)


def test_multi_observable_scoring():
    # steane has k=1 here, but the multi-observable path must accept k>1;
    # exercised on BB in the slow suite. Sanity: distance ints returned.
    code, raw = surface_css(3)
    dz, dx = score_assignment(code, uniform_assignment(raw, N, N), 3, 4)
    assert isinstance(dz, int) and isinstance(dx, int)
    assert min(dz, dx) == 2
