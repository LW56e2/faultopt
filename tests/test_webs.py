"""Pauli-web engine: closed-fragment agreement with declared structure,
open-fragment detector discovery, hook orientation separation."""

from collections import defaultdict

from codes import build_code
from webs import (declared_detectors_and_obs, in_span, orientation, rank,
                  se_fragment)

N = ("NW", "SW", "NE", "SE")
Z = ("NW", "NE", "SW", "SE")


def closed_min_weight(fr, stim_masks, obs):
    sigs = []
    for op_idx, paulis, lab in fr.mechs:
        flips, _ = fr.propagate(op_idx, paulis)
        det = 0
        for j, m in enumerate(stim_masks):
            if bin(flips & m).count("1") & 1:
                det |= 1 << j
        sigs.append((det, bin(flips & obs).count("1") & 1))
    if any(det == 0 and lf for det, lf in sigs):
        return 1
    groups = defaultdict(set)
    for det, lf in sigs:
        groups[det].add(lf)
    if any(v == {0, 1} for v in groups.values()):
        return 2
    return 3  # ">= 3"


def test_closed_web_space_matches_declared():
    checks = build_code(3)
    for sched in ((N, Z), (N, N)):
        fr = se_fragment(3, checks, *sched, 3, closed_basis="Z")
        _, dets = fr.detector_basis()
        incl = [m for _, m in dets]
        stim_masks, obs = declared_detectors_and_obs(3, checks, 3, "Z")
        assert rank(incl) == len(stim_masks) + 1 == 25
        assert all(in_span(m, incl) for m in stim_masks)
        assert in_span(obs, incl)


def test_closed_schedule_separation():
    checks = build_code(3)
    stim_masks, obs = declared_detectors_and_obs(3, checks, 3, "Z")
    good = se_fragment(3, checks, N, Z, 3, closed_basis="Z")
    bad = se_fragment(3, checks, N, N, 3, closed_basis="Z")
    assert closed_min_weight(good, stim_masks, obs) >= 3
    assert closed_min_weight(bad, stim_masks, obs) == 2


def test_open_window_detectors_and_hooks():
    checks = build_code(3)
    for sched, want_vertical in (((N, Z), False), ((N, N), True)):
        fr = se_fragment(3, checks, *sched, rounds=2, closed_basis=None,
                         other_noise=False)
        _, dets = fr.detector_basis()
        assert rank([m for _, m in dets]) == len(checks) == 8
        vertical = 0
        for _, _, residual, _ in fr.mech_signature(dets):
            if orientation(residual, 3) == "VERTICAL":
                vertical += 1
        assert (vertical > 0) == want_vertical
