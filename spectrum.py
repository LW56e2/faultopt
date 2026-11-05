"""TOOL: the open-fragment scorer ("webs" grader backend).

Independent signature propagation and bounded search:

  fragment_from_layers   any gate-layer candidate -> a Fragment (open or
                         closed at either end), with the SAME fault
                         mechanisms as scoring.circuit_from_gates.
  declared_parities      the composite detector/observable inclusion masks
                         for any CSS code (mirrors scoring.py's detectors).
  signatures             per fault mechanism, one bit-vector: which declared
                         parities it flips (via Pauli propagation) -- plus,
                         for open windows, its residual boundary Pauli
                         reduced to a spacetime-equivalence class
                         (mod the boundary actions of the fragment's webs).
  min_weight             smallest number of mechanisms whose combined
                         signature flips a logical and no detector
                         (meet-in-the-middle over deduplicated signatures;
                         exact up to the cutoff).
  window_table           the truncated fault spectrum of an open window:
                         {(local parity flips, residual class): min weight}.
  dominates              pointwise comparison of truncated spectrum tables
                         for spec-equal fragments.
  compose_cut            glue two windows: window-1 residuals are pushed
                         through window 2 (its linear response), signatures
                         are combined, and the composite distance is read
                         off -- the "cut test".
  WebGrader              grader.py backend ("webs"): closed scoring through
                         the web/propagation engine only -- no stim anywhere
                         in the scoring path.

Cutoff semantics: distances are exact when below the cutoff; a returned
value equal to the cutoff means ">= cutoff". Supported cutoff <= 6
(the weight-5 check is O(D^3) over distinct signatures -- see min_weight).
"""

import itertools
from collections import defaultdict

from f2 import in_span, reduce_mod, rref
from grader import register
from webs import PAULIS2, Fragment, _P

# ---------------------------------------------------------------------------
# Fragment construction from gate layers (mirrors scoring.circuit_from_gates)
# ---------------------------------------------------------------------------


def fragment_from_layers(code, layers, rounds, init_basis, final_basis,
                         anc_init_noise=True, noise=True):
    """Build a Fragment for `rounds` rounds of the given gate layers.

    init_basis / final_basis: "Z"/"X" to reset / measure the data qubits
    (closed end), None to leave them as open legs. Ancillas are always
    reset at the start and measure-reset each round. Fault mechanisms
    mirror scoring.circuit_from_gates: init flips, per-round data
    depolarization, 15 two-qubit Paulis per 2q gate (3 per 1q gate),
    measurement flips, final measurement flips."""
    fr = Fragment()
    data = list(range(code.n))
    zanc = [a for k, _, a in code.checks if k == "Z"]
    xanc = [a for k, _, a in code.checks if k == "X"]

    if init_basis:
        for q in data:
            fr.reset(q, init_basis)
            if noise:
                fr.add_mech({q: (1, 0) if init_basis == "Z" else (0, 1)},
                            f"init flip q{q}")
    for q in zanc:
        fr.reset(q, "Z")
        if noise and anc_init_noise:
            fr.add_mech({q: (1, 0)}, f"init flip anc{q}")
    for q in xanc:
        fr.reset(q, "X")
        if noise and anc_init_noise:
            fr.add_mech({q: (0, 1)}, f"init flip anc{q}")

    for t in range(rounds):
        for q in data:
            fr._get(q)
        if noise:
            for q in data:
                for p in _P:
                    fr.add_mech({q: p}, f"idle {_P[p]} q{q} r{t}")
        for layer in layers:
            for name, qs in layer:
                if name == "CX":
                    fr.cx(*qs)
                elif name == "CZ":
                    fr.cz(*qs)
                elif name == "H":
                    fr.h(qs[0])
                elif name == "S":
                    fr.s_gate(qs[0])
                else:
                    raise ValueError(f"unknown gate {name}")
                if not noise:
                    continue
                if len(qs) == 2:
                    for x1, z1, x2, z2 in PAULIS2:
                        fr.add_mech(
                            {qs[0]: (x1, z1), qs[1]: (x2, z2)},
                            f"{name}{qs} r{t} "
                            f"{_P.get((x1, z1), 'I')}{_P.get((x2, z2), 'I')}")
                else:
                    for p in _P:
                        fr.add_mech({qs[0]: p}, f"{name}{qs} r{t} {_P[p]}")
        for q in zanc:
            if noise:
                fr.add_mech({q: (1, 0)}, f"meas flip anc{q} r{t}")
            fr.measure(q, "Z", f"anc{q}:r{t}", reset=True)
        for q in xanc:
            if noise:
                fr.add_mech({q: (0, 1)}, f"meas flip anc{q} r{t}")
            fr.measure(q, "X", f"anc{q}:r{t}", reset=True)

    if final_basis:
        for q in data:
            if noise:
                fr.add_mech({q: (1, 0) if final_basis == "Z" else (0, 1)},
                            f"final meas flip q{q}")
            fr.measure(q, final_basis, f"data{q}", reset=False)
    return fr


def declared_parities(code, rounds, basis):
    """Detector and observable inclusion masks over the measurement record
    (ancillas per round in zanc+xanc order, then data), mirroring
    scoring.circuit_from_gates exactly. Returns (det_masks, obs_masks)."""
    zanc = [a for k, _, a in code.checks if k == "Z"]
    xanc = [a for k, _, a in code.checks if k == "X"]
    order = zanc + xanc
    nm = len(order)
    pos = {q: i for i, q in enumerate(order)}
    base = rounds * nm

    dets = []
    first = zanc if basis == "Z" else xanc
    for q in first:
        dets.append(1 << pos[q])
    for t in range(1, rounds):
        for q in order:
            dets.append((1 << (t * nm + pos[q])) |
                        (1 << ((t - 1) * nm + pos[q])))
    for kind, sup, anc in code.checks:
        if kind != basis:
            continue
        m = 1 << ((rounds - 1) * nm + pos[anc])
        for q in sup:
            m |= 1 << (base + q)
        dets.append(m)
    logicals = code.logicals_z if basis == "Z" else code.logicals_x
    obs = []
    for sup in logicals:
        m = 0
        for q in sup:
            m |= 1 << (base + q)
        obs.append(m)
    return dets, obs


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def local_flips(fr):
    """[(measurement-flip int, residual {q:(x,z)}, label)] per mechanism."""
    out = []
    for op_idx, paulis, label in fr.mechs:
        flips, residual = fr.propagate(op_idx, paulis)
        out.append((flips, residual, label))
    return out


def bits_against(flips, masks):
    sig = 0
    for i, m in enumerate(masks):
        if bin(flips & m).count("1") & 1:
            sig |= 1 << i
    return sig


def closed_signatures(fr, det_masks, obs_masks):
    """Signatures of a closed fragment: bit i flips declared parity i.
    Layout: det bits low, obs bits high."""
    masks = det_masks + obs_masks
    sigs, labels = [], {}
    for flips, residual, label in local_flips(fr):
        assert not residual, f"closed fragment leaked residual: {label}"
        s = bits_against(flips, masks)
        sigs.append(s)
        labels.setdefault(s, label)
    return sigs, labels


# ---------------------------------------------------------------------------
# Min-weight undetectable logical fault (meet-in-the-middle, exact < cutoff)
# ---------------------------------------------------------------------------


def min_weight(sigs, n_det, cutoff=4):
    """Smallest k such that some k signatures XOR to (det = 0, obs != 0).
    Exact for results < cutoff; returns cutoff if none found (">= cutoff").
    Signatures are deduplicated first (two mechanisms with equal signatures
    are interchangeable). Supported cutoff <= 6. Cost over D distinct
    signatures: O(D^2) time/memory for cutoff 4-5; cutoff 6 adds an
    O(D^3)-time weight-5 sweep (fine for D in the hundreds -- Steane
    scale; BB-scale D needs a compiled/parallel kernel first)."""
    if cutoff > 6:
        raise ValueError("cutoff > 6 not implemented")
    detmask = (1 << n_det) - 1
    S = sorted(set(sigs))
    if cutoff > 1 and any(s & detmask == 0 and s >> n_det for s in S):
        return 1
    bucket = defaultdict(set)          # det part -> set of obs parts
    for s in S:
        bucket[s & detmask].add(s >> n_det)
    if cutoff > 2 and any(len(v) >= 2 for v in bucket.values()):
        return 2
    if cutoff <= 3:
        return cutoff
    pair = defaultdict(set)            # det part of s1^s2 -> obs parts
    for s1, s2 in itertools.combinations(S, 2):
        x = s1 ^ s2
        pair[x & detmask].add(x >> n_det)
    if any(o != (s >> n_det)
           for s in S for o in pair.get(s & detmask, ())):
        return 3
    if cutoff > 4 and any(len(v) >= 2 for v in pair.values()):
        return 4
    if cutoff <= 5:
        return cutoff
    # weight 5 = triple vs pair meet-in-the-middle. Any overlap between
    # the triple and the matched pair would cancel to a weight<=3 witness,
    # which was excluded above -- so a hit here is a genuine 5-set.
    for s1, s2, s3 in itertools.combinations(S, 3):
        x = s1 ^ s2 ^ s3
        obs = pair.get(x & detmask)
        if obs and (len(obs) > 1 or (x >> n_det) not in obs):
            return 5
    return cutoff


def find_witness(sigs_labels, n_det, weight):
    """Recover one witness (list of labels) for a result of min_weight."""
    detmask = (1 << n_det) - 1
    items = list(sigs_labels.items())    # (sig, label), deduplicated
    if weight == 1:
        for s, lab in items:
            if s & detmask == 0 and s >> n_det:
                return [lab]
    if weight == 2:
        for (s1, l1), (s2, l2) in itertools.combinations(items, 2):
            x = s1 ^ s2
            if x & detmask == 0 and x >> n_det:
                return [l1, l2]
    if weight == 3:
        by_det = defaultdict(list)
        for s, lab in items:
            by_det[s & detmask].append((s, lab))
        for (s1, l1), (s2, l2) in itertools.combinations(items, 2):
            x = s1 ^ s2
            for s3, l3 in by_det.get(x & detmask, ()):
                if (x ^ s3) >> n_det and len({l1, l2, l3}) == 3:
                    return [l1, l2, l3]
    return None


# ---------------------------------------------------------------------------
# Open windows: residual classes, spectrum tables, domination
# ---------------------------------------------------------------------------


def cut_action_basis(fr, cut_qubits):
    """RREF pivots of the boundary actions of the fragment's webs on the
    cut segments -- the subgroup of residuals equivalent to doing nothing
    (spacetime equivalence, Prop. 4.9)."""
    segs = {q: fr.cur[q] for q in cut_qubits}
    order = sorted(cut_qubits)
    actions = []
    for w in fr.web_basis():
        m = 0
        for i, q in enumerate(order):
            s = segs[q]
            if (w >> (2 * s)) & 1:          # x bit
                m |= 1 << (2 * i)
            if (w >> (2 * s + 1)) & 1:      # z bit
                m |= 1 << (2 * i + 1)
        if m:
            actions.append(m)
    return rref(actions), order


def residual_mask(residual, order):
    m = 0
    for i, q in enumerate(order):
        x, z = residual.get(q, (0, 0))
        if x:
            m |= 1 << (2 * i)
        if z:
            m |= 1 << (2 * i + 1)
    return m


def window_signatures(fr, local_masks, cut_qubits):
    """Per mechanism: (local parity bits | residual class bits, label).
    Residuals are reduced modulo the fragment's own web boundary actions,
    so spacetime-equivalent faults share a signature."""
    piv, order = cut_action_basis(fr, cut_qubits)
    n_local = len(local_masks)
    out = []
    for flips, residual, label in local_flips(fr):
        rc = reduce_mod(residual_mask(residual, order), piv)
        out.append(((bits_against(flips, local_masks)) | (rc << n_local),
                    label))
    return out, piv, order


def spectrum_table(sigs, cutoff=3):
    """Truncated fault spectrum {signature: min weight} from single
    mechanisms and pairs (exact for weights < cutoff <= 3)."""
    if cutoff > 3:
        raise ValueError("table cutoff > 3 not implemented")
    T = {0: 0}
    S = sorted(set(sigs))
    for s in S:
        if s:
            T[s] = 1
    if cutoff > 2:
        for s1, s2 in itertools.combinations(S, 2):
            x = s1 ^ s2
            if x and x not in T:
                T[x] = 2
    return T


def restrict_undetectable(T, internal_mask):
    """Keep only keys whose window-internal detector bits are zero -- the
    faults fault-boundedness actually constrains (detected faults are free
    of obligations, Def. 2.10)."""
    return {k: v for k, v in T.items() if not (k & internal_mask)}


def dominates(TA, TB, cutoff=3):
    """Is A at least as protected as B (up to the cutoff)?
    True iff every nonzero effect is at least as expensive in A:
    S_A(k) >= S_B(k) for all k (missing entries read as >= cutoff).
    Returns (bool, counterexample key or None)."""
    for k in set(TA) | set(TB):
        if k == 0:
            continue
        if TA.get(k, cutoff) < TB.get(k, cutoff):
            return False, k
    return True, None


# ---------------------------------------------------------------------------
# The cut test: glue two windows and read off the composite distance
# ---------------------------------------------------------------------------


def compose_cut(code, fr1, fr2, n1_meas, det_masks, obs_masks, cutoff=4):
    """Composite min-weight undetectable logical fault of fr2 . fr1.

    fr1's residual at the cut is pushed through fr2 (linear response,
    computed by propagation from fr2's open input) and folded into fr1's
    composite signature; fr2's mechanisms are shifted into place; then the
    composite is scored exactly like a closed fragment. det/obs masks are
    the COMPOSITE declared parities; n1_meas = number of measurements
    belonging to fr1."""
    masks = det_masks + obs_masks
    resp_cache = {}

    def response(residual):
        key = tuple(sorted((q, xz) for q, xz in residual.items()))
        if key not in resp_cache:
            flips2, res2 = fr2.propagate(-1, dict(residual))
            assert not res2, "window 2 must be closed at the far end"
            resp_cache[key] = flips2
        return resp_cache[key]

    sigs, labels = [], {}
    for flips, residual, label in local_flips(fr1):
        comp = flips | (response(residual) << n1_meas)
        s = bits_against(comp, masks)
        sigs.append(s)
        labels.setdefault(s, f"[W1] {label}")
    for flips, residual, label in local_flips(fr2):
        assert not residual
        s = bits_against(flips << n1_meas, masks)
        sigs.append(s)
        labels.setdefault(s, f"[W2] {label}")
    w = min_weight(sigs, len(det_masks), cutoff)
    return w, sigs, labels


# ---------------------------------------------------------------------------
# The grader backend
# ---------------------------------------------------------------------------


class WebGrader:
    """Closed grading through the web/propagation engine (no stim in the
    scoring path). Distances are exact below `cutoff`; a returned value
    equal to cutoff means ">= cutoff"."""

    name = "webs"

    def __init__(self, cutoff=4):
        self.cutoff = cutoff

    def measures_code(self, code, layers, rounds=2):
        fr = fragment_from_layers(code, layers, rounds, "Z", "Z",
                                  noise=False)
        det_masks, _ = declared_parities(code, rounds, "Z")
        _, dets = fr.detector_basis()
        incl = [m for _, m in dets]
        return all(in_span(m, incl) for m in det_masks)

    def _one_basis(self, code, layers, rounds, basis):
        fr = fragment_from_layers(code, layers, rounds, basis, basis)
        det_masks, obs_masks = declared_parities(code, rounds, basis)
        sigs, labels = closed_signatures(fr, det_masks, obs_masks)
        return (min_weight(sigs, len(det_masks), self.cutoff), labels,
                len(det_masks), sigs, fr)

    def score(self, code, layers, rounds):
        if not self.measures_code(code, layers):
            raise ValueError("candidate does not measure the code "
                             "(declared detectors not in web span)")
        dz = self._one_basis(code, layers, rounds, "Z")[0]
        dx = self._one_basis(code, layers, rounds, "X")[0]
        return dz, dx

    def explain(self, code, layers, rounds):
        z = self._one_basis(code, layers, rounds, "Z")
        x = self._one_basis(code, layers, rounds, "X")
        basis, (w, labels, nd, sigs, fr) = (("Z", z) if z[0] <= x[0]
                                            else ("X", x))
        witness = find_witness(labels, nd, w)
        if witness is None and w < self.cutoff:
            # deeper witnesses (weight 4-5) via the CEGIS extractor
            from cegis import witness_indices
            idxs = witness_indices(sigs, nd, w)
            if idxs is not None:
                witness = [fr.mechs[i][2] for i in idxs]
        return basis, witness or ["(no witness found below cutoff)"]


register("webs", WebGrader)


# ---------------------------------------------------------------------------
# Beyond SE: state-preparation gadgets (open OUTPUT legs)
# ---------------------------------------------------------------------------


def prep_fragment(n_data, layers, init_basis, ancillas, noise=True):
    """Fragment for a preparation gadget: data qubits reset per
    init_basis[q] ("Z"/"X"), gate layers applied once, verification
    ancillas (reset |0>) measured in Z at the end, data legs left OPEN.
    Mechanisms mirror the usual noise model."""
    fr = Fragment()
    for q in range(n_data):
        b = init_basis.get(q, "Z")
        fr.reset(q, b)
        if noise:
            fr.add_mech({q: (1, 0) if b == "Z" else (0, 1)},
                        f"init flip q{q}")
    for a in ancillas:
        fr.reset(a, "Z")
        if noise:
            fr.add_mech({a: (1, 0)}, f"init flip anc{a}")
    for layer in layers:
        for name, qs in layer:
            if name == "CX":
                fr.cx(*qs)
            elif name == "CZ":
                fr.cz(*qs)
            elif name == "H":
                fr.h(qs[0])
            elif name == "S":
                fr.s_gate(qs[0])
            else:
                raise ValueError(name)
            if not noise:
                continue
            if len(qs) == 2:
                for x1, z1, x2, z2 in PAULIS2:
                    fr.add_mech({qs[0]: (x1, z1), qs[1]: (x2, z2)},
                                f"{name}{qs} "
                                f"{_P.get((x1, z1), 'I')}"
                                f"{_P.get((x2, z2), 'I')}")
            else:
                for p in _P:
                    fr.add_mech({qs[0]: p}, f"{name}{qs} {_P[p]}")
    for a in ancillas:
        if noise:
            fr.add_mech({a: (1, 0)}, f"meas flip anc{a}")
        fr.measure(a, "Z", f"anc{a}", reset=False)
    return fr


def coset_min_weight(mask, basis_rows, max_dim=16):
    """Minimum qubit-weight over mask * span(basis_rows) -- the true
    weight of a residual's spacetime-equivalence class. Enumerates the
    span (guarded by max_dim)."""
    rows = [r for r in basis_rows if r]
    if len(rows) > max_dim:
        raise ValueError(f"coset dimension {len(rows)} > {max_dim}")

    def wt(m):
        w = 0
        while m:
            if m & 3:
                w += 1
            m >>= 2
        return w

    best = wt(mask)
    for pick in range(1, 1 << len(rows)):
        v = mask
        p = pick
        i = 0
        while p:
            if p & 1:
                v ^= rows[i]
            p >>= 1
            i += 1
        best = min(best, wt(v))
    return best


def prep_ft_report(fr, n_data):
    """Fault-tolerance report for a preparation fragment, distance-3
    criterion: every single fault must be either detected (flips a
    verification detector) or leave a residual whose spacetime-equivalence
    class contains a weight <= 1 representative. Returns a dict with the
    violating mechanisms."""
    _, dets = fr.detector_basis()
    piv, order = cut_action_basis(fr, list(range(n_data)))
    rows = list(piv.values())
    sigs = fr.mech_signature(dets)
    violations = []
    detected = clean = 0
    for dmask, flips, residual, label in sigs:
        if dmask:
            detected += 1
            continue
        w = coset_min_weight(residual_mask(residual, order), rows)
        if w >= 2:
            violations.append({"mechanism": label, "class_weight": w})
        else:
            clean += 1
    return {"mechanisms": len(sigs), "detected": detected,
            "harmless": clean, "violations": violations,
            "fault_tolerant_d3": not violations}
