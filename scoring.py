"""Stim-based memory-circuit construction and logical-fault witness search.

Builds noisy memory experiments for a scheduled syndrome-extraction circuit
and searches for undetectable logical-fault witnesses via Stim. The search
is heuristic: a returned witness supplies an upper bound. Use the independent
signature engine for completed lower-bound calculations.

Two circuit builders:
  memory_circuit        uniform corner-permutation schedules (surface code,
                        includes QUBIT_COORDS so fault locations are readable)
  memory_circuit_assign generic per-check edge->layer assignment (any CSS)

Noise model (fault locations only matter for distance): DEPOLARIZE2 on every
CNOT (hooks live here), reset/measurement flips, data depolarization per
round, final measurement flips.
"""

import stim

from codes import build_code, layer_gates


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def memory_circuit(d, checks, sched_z, sched_x, basis, rounds, p=1e-3):
    """d-round memory experiment in `basis` ("Z"/"X"), uniform schedules."""
    data = list(range(d * d))
    zanc = [ch.anc for ch in checks if ch.kind == "Z"]
    xanc = [ch.anc for ch in checks if ch.kind == "X"]
    meas_order = zanc + xanc          # per-round measurement record layout
    mpos = {q: i for i, q in enumerate(meas_order)}
    nm = len(meas_order)
    nd = len(data)

    c = stim.Circuit()
    for r in range(d):
        for cc in range(d):
            c.append("QUBIT_COORDS", [r * d + cc], [2 * cc + 1, 2 * r + 1])
    for ch in checks:
        c.append("QUBIT_COORDS", [ch.anc], [2 * ch.c, 2 * ch.r])

    if basis == "Z":
        c.append("R", data)
        c.append("X_ERROR", data, p)
    else:
        c.append("RX", data)
        c.append("Z_ERROR", data, p)
    c.append("R", zanc)
    c.append("X_ERROR", zanc, p)
    c.append("RX", xanc)
    c.append("Z_ERROR", xanc, p)

    for t in range(rounds):
        c.append("TICK")
        c.append("DEPOLARIZE1", data, p)
        for layer in range(4):
            g = layer_gates(checks, sched_z, sched_x, layer)
            targets = [q for pair in g for q in pair]
            if targets:
                c.append("CX", targets)
                c.append("DEPOLARIZE2", targets, p)
            c.append("TICK")
        c.append("MR", zanc, p)
        c.append("MRX", xanc, p)

        if t == 0:
            first = zanc if basis == "Z" else xanc
            for q in first:
                c.append("DETECTOR", [stim.target_rec(-nm + mpos[q])])
        else:
            for q in meas_order:
                c.append("DETECTOR", [stim.target_rec(-nm + mpos[q]),
                                      stim.target_rec(-2 * nm + mpos[q])])

    if basis == "Z":
        c.append("X_ERROR", data, p)
        c.append("M", data)
    else:
        c.append("Z_ERROR", data, p)
        c.append("MX", data)

    for ch in checks:
        if ch.kind != basis:
            continue
        recs = [stim.target_rec(-nd + q) for _, q in ch.data]
        recs.append(stim.target_rec(-nd - nm + mpos[ch.anc]))
        c.append("DETECTOR", recs)

    if basis == "Z":
        obs = [stim.target_rec(-nd + cc) for cc in range(d)]          # top row
    else:
        obs = [stim.target_rec(-nd + r * d) for r in range(d)]        # left col
    c.append("OBSERVABLE_INCLUDE", obs, 0)
    return c


def circuit_from_gates(code, layers, basis, rounds, p=1e-3):
    """Memory experiment from explicit gate layers, for any CSS code.

    layers: [[(name, qubits), ...], ...] with name in CX/CZ/H/S -- the
    output format of the synthesizer, so full-Clifford candidates are
    scorable, not just CNOT schedules. Noise: DEPOLARIZE2 after every
    2-qubit gate, DEPOLARIZE1 after every 1-qubit gate, plus the usual
    reset/measurement flips and per-round data depolarization."""
    data = list(range(code.n))
    zanc = [a for k, _, a in code.checks if k == "Z"]
    xanc = [a for k, _, a in code.checks if k == "X"]
    meas_order = zanc + xanc
    mpos = {q: i for i, q in enumerate(meas_order)}
    nm, nd = len(meas_order), len(data)

    c = stim.Circuit()
    for q, xy in sorted(getattr(code, "coords", {}).items()):
        c.append("QUBIT_COORDS", [q], list(xy))
    if basis == "Z":
        c.append("R", data)
        c.append("X_ERROR", data, p)
    else:
        c.append("RX", data)
        c.append("Z_ERROR", data, p)
    c.append("R", zanc)
    c.append("X_ERROR", zanc, p)
    c.append("RX", xanc)
    c.append("Z_ERROR", xanc, p)

    for t in range(rounds):
        c.append("TICK")
        c.append("DEPOLARIZE1", data, p)
        for layer in layers:
            two_q, one_q = [], []
            for name, qs in layer:
                c.append(name if name != "CZ" else "CZ", list(qs))
                (two_q if len(qs) == 2 else one_q).extend(qs)
            if two_q:
                c.append("DEPOLARIZE2", two_q, p)
            if one_q:
                c.append("DEPOLARIZE1", one_q, p)
            c.append("TICK")
        c.append("MR", zanc, p)
        c.append("MRX", xanc, p)
        first = zanc if basis == "Z" else xanc
        if t == 0:
            for q in first:
                c.append("DETECTOR", [stim.target_rec(-nm + mpos[q])])
        else:
            for q in meas_order:
                c.append("DETECTOR", [stim.target_rec(-nm + mpos[q]),
                                      stim.target_rec(-2 * nm + mpos[q])])

    if basis == "Z":
        c.append("X_ERROR", data, p)
        c.append("M", data)
    else:
        c.append("Z_ERROR", data, p)
        c.append("MX", data)
    for kind, sup, anc in code.checks:
        if kind != basis:
            continue
        recs = [stim.target_rec(-nd + q) for q in sup]
        recs.append(stim.target_rec(-nd - nm + mpos[anc]))
        c.append("DETECTOR", recs)
    logicals = code.logicals_z if basis == "Z" else code.logicals_x
    for oi, obs in enumerate(logicals):
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(-nd + q) for q in obs], oi)
    return c


def layers_from_assignment(code, assignment, T):
    """Per-check edge->layer schedule, as explicit CX gate layers."""
    layers = [[] for _ in range(T)]
    for (i, q), lay in assignment.items():
        kind, _, anc = code.checks[i]
        layers[lay].append(("CX", (q, anc) if kind == "Z" else (anc, q)))
    return layers


def memory_circuit_assign(code, assignment, basis, rounds, T, p=1e-3):
    """Memory experiment from a per-check CNOT schedule (any CSS code)."""
    return circuit_from_gates(code, layers_from_assignment(code, assignment, T),
                              basis, rounds, p)


# ---------------------------------------------------------------------------
# Distance scoring
# ---------------------------------------------------------------------------


def circuit_distance(circ):
    """Heuristic logical-fault search: return (witness weight, witness).

    The legacy function name is retained; this does not certify a lower bound.
    """
    errs = circ.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    return len(errs), errs


def score_schedule(args):
    """Sweep worker for uniform schedules (multiprocessing-friendly).

    Returns d_eff = None when the schedule does not measure the code's
    stabilizers (corrupted interleaving -> non-deterministic detectors)."""
    d, rounds, sched_z, sched_x = args
    checks = build_code(d)
    try:
        dz, _ = circuit_distance(
            memory_circuit(d, checks, sched_z, sched_x, "Z", rounds))
        dx, _ = circuit_distance(
            memory_circuit(d, checks, sched_z, sched_x, "X", rounds))
    except ValueError:
        return {"sched_z": sched_z, "sched_x": sched_x, "d_z": None,
                "d_x": None, "d_eff": None}
    return {"sched_z": sched_z, "sched_x": sched_x, "d_z": dz, "d_x": dx,
            "d_eff": min(dz, dx)}


def score_assignment(code, assignment, rounds, T):
    """(d_Z, d_X) for a per-check schedule on any CSS code."""
    return score_gates(code, layers_from_assignment(code, assignment, T),
                       rounds)


def score_gates(code, layers, rounds):
    """(d_Z, d_X) for explicit gate layers on any CSS code.

    Raises ValueError if the candidate does not measure the code (the spec
    gap becomes visible here rather than passing silently)."""
    dz, _ = circuit_distance(circuit_from_gates(code, layers, "Z", rounds))
    dx, _ = circuit_distance(circuit_from_gates(code, layers, "X", rounds))
    return dz, dx


def deterministic(code, assignment, T):
    """Does the schedule measure the code (deterministic detectors)?"""
    try:
        memory_circuit_assign(code, assignment, "Z", 2, T,
                              p=0).detector_error_model()
        return True
    except ValueError:
        return False


def prepares_state(code, layers, init_basis, n_anc=0):
    """Tableau check (unitary candidates, n_anc=0): after resetting data
    per init_basis and applying the layers, is the state stabilized by
    every check and every logical-Z representative (up to sign)?"""
    import stim
    n_total = code.n + n_anc
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n_total)
    for q in range(code.n):
        if init_basis.get(q, "Z") == "X":
            sim.h(q)
    for layer in layers:
        for name, qs in layer:
            getattr(sim, {"CX": "cx", "CZ": "cz", "H": "h",
                          "S": "s"}[name])(*qs)
    targets = ([(k, s) for k, s, _ in code.checks]
               + [("Z", s) for s in code.logicals_z])
    for kind, sup in targets:
        ps = ["_"] * n_total
        for q in sup:
            ps[q] = "X" if kind == "X" else "Z"
        val = sim.peek_observable_expectation(
            stim.PauliString("".join(ps)))
        if val == 0:
            return False        # not stabilized (random outcome)
    return True
