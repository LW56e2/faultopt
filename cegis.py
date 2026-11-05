"""TOOL: the CEGIS feedback arrow (counterexample-guided schedule search).

Closes the big-loop gap of docs/METHODS.md. Blind
blocking-clause enumeration removes ONE candidate per solve; here every
badly scored candidate is turned into a GENERALIZED blocking clause that
removes every schedule sharing the sub-assignment responsible for the
failure:

    synth = make_se_synth(code, T, "sched", ...)
    while SAT:
        candidate = model
        score both bases with the web/propagation engine (spectrum.py)
        if d_eff >= d_target:  winner (return it; or, in exhaust mode,
                               block exactly this candidate and continue)
        else:                  extract a minimal witness (fault mechanisms,
                               weight < d_target), compute its CONE, and
                               block the cone's edge->layer pattern
    UNSAT => PROOF: no depth-T schedule beyond the returned winners reaches
             d_eff >= d_target (under scoring.circuit_from_gates's noise
             model at the given number of rounds; with tie_classes, the
             proof covers orbit-symmetric schedules only).

The cone of a witness S, computed on the failing candidate A:
  * propagate every mechanism of S to the end of A's fragment and collect
    every qubit (data or ancilla) that ever carries a nonzero Pauli --
    the involved set Q(S);
  * E(S) = every Tanner edge incident to a qubit of Q(S) (all edges of a
    check whose ancilla is involved; all edges (j, q) of an involved data
    qubit q);
  * the clause forbids firing every edge of E(S) at its layer in A.

Soundness (why final UNSAT is a proof, not a heuristic exhaustion): let B
be any schedule that fires every edge of E(S) at the same layer as A. In
the scheduling library each Tanner edge fires exactly once and the
non-scheduled skeleton (resets, measure-resets, idle/init/measurement
fault locations, declared detector/observable parities) is identical
across candidates, so S transports verbatim to B. By induction over ops,
S's propagation in B is layer-by-layer identical on Q(S): every gate that
can touch an involved qubit IS an edge of E(S) (fired at the same layer),
gates outside E(S) never touch a qubit carrying a nonzero Pauli, and
same-layer gates act on disjoint qubits (parallelism) so within-layer
order is irrelevant. Hence S flips the same measurements, the same
declared detectors (none) and the same observables (some) on B, and B
scores < d_target too. Every schedule a clause excludes is provably bad.

Scope: the SCHEDULING library only -- the transport argument above needs
"the same gate" to be well-defined across candidates (exactly-once Tanner
edges). Richer libraries (clifford, prep) need their own transport
argument and are refused. d_target <= 6: witnesses below the cutoff have
weight <= 5, and the min-weight engine's cutoff tops out at 6 (its
weight-5 sweep is O(D^3) over distinct signatures -- cheap at Steane
scale, BB scale needs a compiled/parallel kernel; docs/REPRODUCIBILITY.md).
The stim backend cannot drive this loop today: its witnesses live in stim
circuit coordinates and would need location parsing to map onto our
mechanisms; the web engine's witnesses ARE our mechanisms (and the two
engines cross-validate, see tests/test_spectrum.py).
"""

import itertools
import sys
import time
from collections import defaultdict

import z3

from spectrum import (closed_signatures, declared_parities,
                      fragment_from_layers, min_weight)
from synth import make_se_synth


# ---------------------------------------------------------------------------
# Schedule bookkeeping
# ---------------------------------------------------------------------------


def assignment_from_layers(code, layers):
    """Inverse of scoring.layers_from_assignment:
    gate layers -> {(check_idx, data_qubit): layer}. Sched-mode candidates
    only (every gate a CX realizing one Tanner edge)."""
    by_anc = {anc: i for i, (_, _, anc) in enumerate(code.checks)}
    assign = {}
    for t, layer in enumerate(layers):
        for name, qs in layer:
            if name != "CX":
                raise ValueError(f"non-CX gate {name}{qs}: not a "
                                 f"scheduling-library candidate")
            anc = [q for q in qs if q >= code.n]
            if len(anc) != 1:
                raise ValueError(f"CX{qs} is not a Tanner-edge CNOT")
            data = qs[0] if qs[1] == anc[0] else qs[1]
            assign[(by_anc[anc[0]], data)] = t
    return assign


# ---------------------------------------------------------------------------
# Witness cone: which schedule decisions a failing fault set depends on
# ---------------------------------------------------------------------------


def propagate_tracking(fr, mech_idx):
    """Like Fragment.propagate for mechanism mech_idx, but also collect
    every qubit that ever carries a nonzero Pauli. Returns
    (flip mask, residual, involved qubits). Agreement of (flips, residual)
    with fr.propagate is asserted in tests to guard against drift."""
    op_idx, paulis, _ = fr.mechs[mech_idx]
    pauli = {q: p for q, p in paulis.items() if p != (0, 0)}
    involved = set(pauli)
    flips = 0
    mcount = sum(1 for i in range(op_idx + 1)
                 if fr.ops[i][0].startswith("M"))
    for i in range(op_idx + 1, len(fr.ops)):
        name, qs = fr.ops[i]
        if name == "CX":
            c, t = qs
            xc, zc = pauli.get(c, (0, 0))
            xt, zt = pauli.get(t, (0, 0))
            pauli[c] = (xc, zc ^ zt)
            pauli[t] = (xt ^ xc, zt)
        elif name == "CZ":
            a, b = qs
            xa, za = pauli.get(a, (0, 0))
            xb, zb = pauli.get(b, (0, 0))
            pauli[a] = (xa, za ^ xb)
            pauli[b] = (xb, zb ^ xa)
        elif name == "H":
            x, z = pauli.get(qs[0], (0, 0))
            pauli[qs[0]] = (z, x)
        elif name == "S":
            x, z = pauli.get(qs[0], (0, 0))
            pauli[qs[0]] = (x, z ^ x)
        elif name.startswith("R"):
            pauli.pop(qs[0], None)
        elif name.startswith("M"):
            q = qs[0]
            x, z = pauli.get(q, (0, 0))
            if x if name[1] == "Z" else z:
                flips |= 1 << mcount
            mcount += 1
            pauli.pop(q, None)
        for q in qs:
            if pauli.get(q, (0, 0)) != (0, 0):
                involved.add(q)
    residual = {q: p for q, p in pauli.items()
                if p != (0, 0) and q not in fr.terminated}
    return flips, residual, involved


def witness_cone(code, fr, mech_indices):
    """E(S): every Tanner edge incident to any qubit involved in the
    propagation of any witness mechanism (see module docstring)."""
    involved = set()
    for i in mech_indices:
        involved |= propagate_tracking(fr, i)[2]
    edges = set()
    for ci, (_, sup, anc) in enumerate(code.checks):
        if anc in involved:
            edges.update((ci, q) for q in sup)
        else:
            edges.update((ci, q) for q in sup if q in involved)
    return sorted(edges)


# ---------------------------------------------------------------------------
# Witness extraction at mechanism-index granularity
# ---------------------------------------------------------------------------


def witness_indices(sigs, n_det, w):
    """One weight-w witness as MECHANISM indices: w mechanisms with
    distinct signatures XORing to (detectors = 0, observables != 0).
    Supports w <= 5, which covers every witness below a min_weight cutoff
    of <= 6. Returns None if no witness exists at exactly weight w.

    Assumes w is the MINIMAL weight (min_weight's answer): witnesses with
    repeated or overlapping sub-combinations would reduce to a smaller
    weight, so signatures may be deduplicated and pair overlaps ignored
    (asserted, not assumed silently)."""
    detmask = (1 << n_det) - 1
    rep = {}                       # signature -> first mechanism index
    for i, s in enumerate(sigs):
        if s:
            rep.setdefault(s, i)
    items = sorted(rep.items())
    if w == 1:
        for s, i in items:
            if not s & detmask and s >> n_det:
                return [i]
        return None
    if w == 2:
        by_det = defaultdict(list)
        for s, i in items:
            by_det[s & detmask].append((s, i))
        for group in by_det.values():
            for (s1, i1), (s2, i2) in itertools.combinations(group, 2):
                if (s1 ^ s2) >> n_det:
                    return [i1, i2]
        return None
    if w == 3:
        by_det = defaultdict(list)
        for s, i in items:
            by_det[s & detmask].append((s, i))
        for (s1, i1), (s2, i2) in itertools.combinations(items, 2):
            x = s1 ^ s2
            for s3, i3 in by_det.get(x & detmask, ()):
                if (x ^ s3) >> n_det:
                    assert len({i1, i2, i3}) == 3, "w=3 not minimal"
                    return [i1, i2, i3]
        return None
    if w == 4:
        first = {}                 # det part of pair XOR -> (obs, i, j)
        for (s1, i1), (s2, i2) in itertools.combinations(items, 2):
            x = s1 ^ s2
            key, obs = x & detmask, x >> n_det
            if key not in first:
                first[key] = (obs, i1, i2)
                continue
            obs0, j1, j2 = first[key]
            if obs != obs0:
                assert not {i1, i2} & {j1, j2}, "w=4 not minimal"
                return [j1, j2, i1, i2]
        return None
    if w == 5:
        pairs = {}                     # det of pair XOR -> {obs: (i, j)}
        for (s1, i1), (s2, i2) in itertools.combinations(items, 2):
            x = s1 ^ s2
            pairs.setdefault(x & detmask, {}).setdefault(
                x >> n_det, (i1, i2))
        for (sa, ia), (sb, ib), (sc, ic) in \
                itertools.combinations(items, 3):
            x = sa ^ sb ^ sc
            group = pairs.get(x & detmask)
            if not group:
                continue
            for obs, (i1, i2) in group.items():
                if obs != x >> n_det:
                    assert not {ia, ib, ic} & {i1, i2}, "w=5 not minimal"
                    return [ia, ib, ic, i1, i2]
        return None
    raise ValueError("witness extraction implemented for weight <= 5 only")


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def cegis_search(code, T, d_target, rounds=3, tie_classes=False,
                 exhaust=False, max_iters=None, verbose=False,
                 kernel=None, procs=1):
    """CEGIS over depth-T scheduling candidates, targeting
    d_eff >= d_target at the given rounds (both bases, webs engine,
    cutoff = d_target so winners carry a certified ">= d_target").

    exhaust=False: stop at the first winner (status "found").
    exhaust=True:  block each winner exactly and keep going; on
                   status "exhausted" the winner list is COMPLETE.

    Returns a dict: status ("found" | "exhausted" | "max_iters"),
    iterations, learned_clauses, winners [{layers, scores}], patterns
    (the learned clauses, for auditing -- see enumerate_matching), log
    (per-iteration scores), seconds, proof (True iff exhausted).
    "exhausted" is a proof in the module-docstring sense; with
    tie_classes it covers orbit-symmetric schedules only."""
    if not 2 <= d_target <= 6:
        raise ValueError("d_target must be in 2..6 (min-weight engine "
                         "cutoff; docs/REPRODUCIBILITY.md). Note target 6 runs "
                         "the O(D^3) weight-5 sweep per candidate -- "
                         "Steane-scale yes, BB-scale needs the HPC kernel")
    t0 = time.time()
    if kernel == "numpy":
        from mim import min_weight_packed
        mw = lambda s, nd, c: min_weight_packed(s, nd, c, procs=procs)
    elif kernel == "gpu":
        from mim_gpu import min_weight_gpu
        mw = lambda s, nd, c: min_weight_gpu(s, nd, c)
    elif kernel is None:
        mw = min_weight
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    synth = make_se_synth(code, T, "sched", tie_classes=tie_classes)
    parities = {b: declared_parities(code, rounds, b) for b in "ZX"}
    winners, patterns, log = [], [], []
    it = 0
    while True:
        if max_iters is not None and it >= max_iters:
            status = "max_iters"
            break
        if not synth.check():
            status = "exhausted"
            break
        it += 1
        layers = synth.model_layers()
        assign = assignment_from_layers(code, layers)
        scores, frs, sig_of = {}, {}, {}
        for basis in "ZX":
            det_masks, obs_masks = parities[basis]
            fr = fragment_from_layers(code, layers, rounds, basis, basis)
            sigs, _ = closed_signatures(fr, det_masks, obs_masks)
            scores[basis] = mw(sigs, len(det_masks), d_target)
            frs[basis], sig_of[basis] = fr, (sigs, len(det_masks))
        d_eff = min(scores.values())
        entry = {"iter": it, "d_z": scores["Z"], "d_x": scores["X"],
                 "winner": d_eff >= d_target}
        if d_eff >= d_target:
            winners.append({"layers": layers,
                            "scores": {b: f">={v}" if v == d_target else v
                                       for b, v in scores.items()}})
            if verbose:
                print(f"[cegis] it={it} winner", file=sys.stderr)
            if not exhaust:
                status = "found"
                log.append(entry)
                break
            synth.solver.add(z3.Not(z3.And(
                [synth.edge_sel[e, t] for e, t in sorted(assign.items())])))
        else:
            blocked_exact = False
            for basis in "ZX":
                if scores[basis] >= d_target:
                    continue
                sigs, n_det = sig_of[basis]
                # weight-5 witness recovery is O(D^3) pure Python; beyond
                # ~3000 distinct signatures fall back to blocking exactly
                # this candidate (sound, just ungeneralized)
                if scores[basis] == 5 and len(set(sigs)) > 3000:
                    if not blocked_exact:
                        synth.solver.add(z3.Not(z3.And(
                            [synth.edge_sel[e, t]
                             for e, t in sorted(assign.items())])))
                        blocked_exact = True
                        entry.setdefault("clauses", []).append(
                            {"basis": basis, "cone_edges": None,
                             "note": "w5-exact-block"})
                    continue
                idxs = witness_indices(sigs, n_det, scores[basis])
                assert idxs is not None, "min_weight promised a witness"
                cone = witness_cone(code, frs[basis], idxs)
                pattern = [(e, assign[e]) for e in cone]
                patterns.append({
                    "basis": basis, "weight": scores[basis],
                    "witness": [frs[basis].mechs[i][2] for i in idxs],
                    "pattern": pattern})
                synth.solver.add(z3.Not(z3.And(
                    [synth.edge_sel[e, t] for e, t in pattern])))
                entry.setdefault("clauses", []).append(
                    {"basis": basis, "cone_edges": len(cone)})
            if verbose:
                sizes = [c["cone_edges"] for c in entry.get("clauses", [])]
                print(f"[cegis] it={it} d_z={scores['Z']} "
                      f"d_x={scores['X']} cones={sizes}", file=sys.stderr)
        log.append(entry)
    return {"status": status, "T": T, "d_target": d_target,
            "rounds": rounds, "tie": tie_classes, "iterations": it,
            "learned_clauses": len(patterns), "winners": winners,
            "patterns": patterns, "log": log, "proof": status == "exhausted",
            "seconds": round(time.time() - t0, 2)}


def enumerate_matching(code, T, pattern, n, tie_classes=False):
    """Up to n distinct schedules satisfying an edge->layer pattern
    (fresh synthesizer, assumption-based). Audit helper: every schedule a
    learned clause excludes must score below the CEGIS target -- the tests
    check that against stim."""
    synth = make_se_synth(code, T, "sched", tie_classes=tie_classes)
    assumptions = [synth.edge_sel[tuple(e), t] for e, t in pattern]
    out = []
    while len(out) < n and synth._decided(synth.solver.check(*assumptions)):
        out.append(synth.model_layers())
        m = synth.solver.model()
        synth.solver.add(z3.Or([
            lit != z3.is_true(m[lit]) for lit in synth.b.values()]))
    return out
