"""TOOL: magic-state injection experiments + the undetected-corruption census.

Injection is never fault tolerant: some single faults corrupt the injected
state without firing any detector.  The figure of merit is therefore not a
distance but a CENSUS -- how many weight-1 (and weight-2) fault mechanisms
are accepted by post-selection yet corrupt the injected logical state --
plus the probability-weighted leading coefficient (p_L ~ c1*p under the
standard location model, given post-selection on zero detection events).

Non-Clifford inputs are scored through CLIFFORD STAND-IN FRAMES
(docs/METHODS.md):

  standard injection (state on a data qubit, first SE round projects it in):
      frame Z = site initialized |0>, observable Z_L
      frame X = site initialized |+>, observable X_L
    A residual anticommutes with Y_L iff it anticommutes with exactly one
    of X_L, Z_L, so the two frames determine the full logical action of
    every residual; a mechanism corrupts a generic injected state iff it
    is accepted and flips the observable in at least one frame.

  hook injection (Gidney arXiv:2302.12292, transcribed to our Z-checks):
    the round-1 rotation exp(-i theta Z/2) on a Z-check ancilla between
    its k-th and (k+1)-th CNOT is the injection knob; the Clifford frames
    theta = 0 ("frameX") and theta = 90 deg = S ("frameY") carry the
    census; theta = 180 deg = Z ("frameZrot") is a VALIDATION-ONLY third
    frame: a measurement-record parity's expectation is A + B cos(theta)
    + C sin(theta), so being constant at 0/90/180 degrees forces B=C=0 --
    i.e. detectors kept by all three frames are deterministic for EVERY
    injected angle, including the non-Clifford T. (For standard injection
    two frames suffice: both circuits are identical, so a record parity's
    Heisenberg image is one fixed Pauli, and determinism in both frames
    forces it to be trivial on the site.)

Experiment shape (both engines, mechanism-for-mechanism parity):
  per-qubit data init (R / RX per init_map)  ->  len(round_layers) noisy
  rounds (per-round gate layers; MR ancillas)  ->  NOISELESS readout: in
  stim, one MPP per check (detector vs its last ancilla record) and one
  MPP for the observable; in the webs engine the data legs stay OPEN and
  the same parities are read off each mechanism's boundary residual.
  The noiseless readout is a scoring device: it contributes no fault
  locations, so the census is exactly about the noisy injection window.

Declared parities are DERIVED, not assumed:
  round-1 detectors     candidates = codes.determined_first_checks;
  round-t>1 detectors   candidates = consecutive-round compares;
  final detectors       check MPP vs last round;
  then every candidate is kept only if a p=0 stim sampling of ALL frames
  shows a constant, frame-independent parity, and the kept set is
  certified by building each frame's p=0 detector error model (stim
  raises on any nondeterministic declared parity).
  The observable per frame is auto-selected: the first of X_L/Y_L/Z_L
  whose determinism is realizable as a Pauli web of the fragment; the
  web's included measurements are folded into OBSERVABLE_INCLUDE (this is
  how e.g. the teleportation correction of hook injection enters).
"""

from collections import defaultdict

import stim

from webs import PAULIS2, Fragment, _P

GOOD_SCHED = (("NW", "SW", "NE", "SE"), ("NW", "NE", "SW", "SE"))


# ---------------------------------------------------------------------------
# Pauli helpers
# ---------------------------------------------------------------------------


def logical_pauli(code, kind):
    """{qubit: (x,z)} for the X_L / Z_L / Y_L representative (Y = X*Z)."""
    if kind == "X":
        return {q: (1, 0) for q in code.logical_x}
    if kind == "Z":
        return {q: (0, 1) for q in code.logical_z}
    p = {q: (1, 0) for q in code.logical_x}
    for q in code.logical_z:
        x, z = p.get(q, (0, 0))
        p[q] = (x, z ^ 1)
    return p


def check_pauli(code, ci):
    kind, sup, _ = code.checks[ci]
    return {q: (1, 0) if kind == "X" else (0, 1) for q in sup}


def anticommutes(residual, pauli):
    t = 0
    for q, (ox, oz) in pauli.items():
        rx, rz = residual.get(q, (0, 0))
        t ^= (rx & oz) ^ (rz & ox)
    return t


def _parity(x):
    return bin(x).count("1") & 1


def mpp_targets(pauli):
    ts = []
    for q in sorted(pauli):
        x, z = pauli[q]
        if not (x or z):
            continue
        if ts:
            ts.append(stim.target_combiner())
        ts.append(stim.target_x(q) if (x, z) == (1, 0) else
                  stim.target_z(q) if (x, z) == (0, 1) else stim.target_y(q))
    return ts


# ---------------------------------------------------------------------------
# Declared parities (shared between the two builders by construction)
# ---------------------------------------------------------------------------


def candidate_detectors(code, rounds):
    """Detector candidate ids: ("one", t, ci) single ancilla record at
    round t (deterministic e.g. for init-determined round-1 checks, for
    first-time-deterministic checks of a grown patch, or for idle
    ancillas), ("cmp", t, ci) consecutive-round compares, ("fin", ci)
    noiseless final check MPP vs the last ancilla record.  The set is
    intentionally over-complete (linear dependencies are harmless):
    multi-frame pruning keeps exactly the parities that are deterministic
    with a frame-independent constant, which for the frame families used
    here is a PROOF of determinism under the real, non-Clifford injection
    (see module docstring).  codes.determined_first_checks predicts the
    kept ("one", 0, ci) entries for plain init-projection rounds and is
    cross-checked in tests."""
    spec = []
    for t in range(rounds):
        for ci in range(len(code.checks)):
            spec.append(("one", t, ci))
    for t in range(1, rounds):
        for ci in range(len(code.checks)):
            spec.append(("cmp", t, ci))
    for ci in range(len(code.checks)):
        spec.append(("fin", ci))
    return spec


def _meas_layout(code):
    zanc = [a for k, _, a in code.checks if k == "Z"]
    xanc = [a for k, _, a in code.checks if k == "X"]
    order = zanc + xanc
    return zanc, xanc, order, {q: i for i, q in enumerate(order)}


def det_recs(code, entry, rounds):
    """Absolute measurement-record indices of one detector entry.
    Layout: rounds*nm ancilla records (zanc then xanc per round), then one
    MPP record per check, then the observable MPP."""
    _, _, order, mpos = _meas_layout(code)
    nm = len(order)
    tag = entry[0]
    if tag == "one":
        _, t, ci = entry
        return [t * nm + mpos[code.checks[ci][2]]]
    if tag == "cmp":
        _, t, ci = entry
        pos = mpos[code.checks[ci][2]]
        return [t * nm + pos, (t - 1) * nm + pos]
    ci = entry[1]
    return [rounds * nm + ci,
            (rounds - 1) * nm + mpos[code.checks[ci][2]]]


# ---------------------------------------------------------------------------
# The two builders (mechanism parity is load-bearing; keep them in lockstep)
# ---------------------------------------------------------------------------


def experiment_circuit(code, round_layers, init_map, det_spec, obs_pauli,
                       fold_recs=(), p=1e-3):
    """stim circuit of the generalized experiment (see module docstring).
    Detectors/observable are appended after all measurements, so record
    arithmetic uses one fixed total."""
    data = list(range(code.n))
    zanc, xanc, order, mpos = _meas_layout(code)
    nm = len(order)
    rounds = len(round_layers)
    nc = len(code.checks)
    total = rounds * nm + nc + 1

    c = stim.Circuit()
    for q, xy in sorted(getattr(code, "coords", {}).items()):
        c.append("QUBIT_COORDS", [q], list(xy))
    zinit = [q for q in data if init_map.get(q, "Z") == "Z"]
    xinit = [q for q in data if init_map.get(q, "Z") == "X"]
    if zinit:
        c.append("R", zinit)
        c.append("X_ERROR", zinit, p)
    if xinit:
        c.append("RX", xinit)
        c.append("Z_ERROR", xinit, p)
    c.append("R", zanc)
    c.append("X_ERROR", zanc, p)
    c.append("RX", xanc)
    c.append("Z_ERROR", xanc, p)

    for layers in round_layers:
        c.append("TICK")
        c.append("DEPOLARIZE1", data, p)
        for layer in layers:
            two_q, one_q = [], []
            for name, qs in layer:
                c.append(name, list(qs))
                (two_q if len(qs) == 2 else one_q).extend(qs)
            if two_q:
                c.append("DEPOLARIZE2", two_q, p)
            if one_q:
                c.append("DEPOLARIZE1", one_q, p)
            c.append("TICK")
        c.append("MR", zanc, p)
        c.append("MRX", xanc, p)

    for ci in range(nc):
        c.append("MPP", mpp_targets(check_pauli(code, ci)))
    c.append("MPP", mpp_targets(obs_pauli))

    for entry in det_spec:
        c.append("DETECTOR", [stim.target_rec(i - total)
                              for i in det_recs(code, entry, rounds)])
    obs = [stim.target_rec(rounds * nm + nc - total)]
    obs += [stim.target_rec(i - total) for i in sorted(fold_recs)]
    c.append("OBSERVABLE_INCLUDE", obs, 0)
    return c


def experiment_fragment(code, round_layers, init_map, noise=True):
    """Webs-engine mirror of experiment_circuit: closed per-qubit init,
    the same noisy rounds, data legs left OPEN (the noiseless readout is
    applied as residual classification, not as fragment ops).
    Returns (fragment, weights) with weights[i] = mechanism i's probability
    as a fraction of the base rate p (1 for flips, 1/3 per 1q-depolarize
    Pauli, 1/15 per 2q-depolarize Pauli)."""
    fr = Fragment()
    weights = []

    def mech(paulis, label, w):
        fr.add_mech(paulis, label)
        weights.append(w)

    data = list(range(code.n))
    zanc, xanc, _, _ = _meas_layout(code)
    for q in data:
        b = init_map.get(q, "Z")
        fr.reset(q, b)
        if noise:
            mech({q: (1, 0) if b == "Z" else (0, 1)}, f"init flip q{q}", 1.0)
    for a in zanc:
        fr.reset(a, "Z")
        if noise:
            mech({a: (1, 0)}, f"init flip anc{a}", 1.0)
    for a in xanc:
        fr.reset(a, "X")
        if noise:
            mech({a: (0, 1)}, f"init flip anc{a}", 1.0)

    for t, layers in enumerate(round_layers):
        for q in data:
            fr._get(q)
        if noise:
            for q in data:
                for pp in _P:
                    mech({q: pp}, f"idle {_P[pp]} q{q} r{t}", 1 / 3)
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
                        mech({qs[0]: (x1, z1), qs[1]: (x2, z2)},
                             f"{name}{qs} r{t} "
                             f"{_P.get((x1, z1), 'I')}"
                             f"{_P.get((x2, z2), 'I')}", 1 / 15)
                else:
                    for pp in _P:
                        mech({qs[0]: pp}, f"{name}{qs} r{t} {_P[pp]}", 1 / 3)
        for q in zanc:
            if noise:
                mech({q: (1, 0)}, f"meas flip anc{q} r{t}", 1.0)
            fr.measure(q, "Z", f"anc{q}:r{t}", reset=True)
        for q in xanc:
            if noise:
                mech({q: (0, 1)}, f"meas flip anc{q} r{t}", 1.0)
            fr.measure(q, "X", f"anc{q}:r{t}", reset=True)
    return fr, weights


# ---------------------------------------------------------------------------
# Observable realization (Pauli-web solve) and per-mechanism signatures
# ---------------------------------------------------------------------------


def observable_fold(fr, code, obs_pauli):
    """Measurement records that make obs_pauli deterministic: solve for a
    web whose boundary action on the open final data legs equals obs_pauli
    (ancilla legs end in fresh resets, where webs can only carry the
    harmless reset-basis label). Returns a bitmask over the fragment's
    measurement indices, or None if no such web exists -- i.e. the
    initialization pattern does not inject a state readable in this basis."""
    order = list(range(code.n))
    segs = {q: fr.cur[q] for q in order}

    def action(w):
        v = 0
        for j, q in enumerate(order):
            s = segs[q]
            if (w >> (2 * s)) & 1:
                v |= 1 << (2 * j)
            if (w >> (2 * s + 1)) & 1:
                v |= 1 << (2 * j + 1)
        return v

    piv = {}
    for w in fr.web_basis():
        v = action(w)
        changed = True
        while changed:
            changed = False
            for b in list(piv):
                if (v >> b) & 1:
                    pv, pw = piv[b]
                    v ^= pv
                    w ^= pw
                    changed = True
        if v:
            piv[v.bit_length() - 1] = (v, w)

    tv = 0
    for j, q in enumerate(order):
        ox, oz = obs_pauli.get(q, (0, 0))
        if ox:
            tv |= 1 << (2 * j)
        if oz:
            tv |= 1 << (2 * j + 1)
    tw = 0
    changed = True
    while changed:
        changed = False
        for b in list(piv):
            if (tv >> b) & 1:
                pv, pw = piv[b]
                tv ^= pv
                tw ^= pw
                changed = True
    if tv:
        return None
    mask = 0
    for i, (var, _) in enumerate(fr.meas):
        if (tw >> var) & 1:
            mask |= 1 << i
    return mask


def frame_signatures(fr, code, det_spec, obs_pauli, fold_mask, rounds):
    """[(detector bits, observable bit, label)] per mechanism, reading the
    SAME declared parities as the stim circuit: ancilla-record bits from
    propagation flips, final-readout bits from the boundary residual."""
    _, _, order, mpos = _meas_layout(code)
    nm = len(order)
    cpaulis = {ci: check_pauli(code, ci) for ci in range(len(code.checks))}
    out = []
    for op_idx, paulis, label in fr.mechs:
        flips, residual = fr.propagate(op_idx, paulis)
        det = 0
        for k, entry in enumerate(det_spec):
            tag = entry[0]
            if tag == "one":
                _, t, ci = entry
                bit = (flips >> (t * nm + mpos[code.checks[ci][2]])) & 1
            elif tag == "cmp":
                _, t, ci = entry
                pos = mpos[code.checks[ci][2]]
                bit = ((flips >> (t * nm + pos))
                       ^ (flips >> ((t - 1) * nm + pos))) & 1
            else:
                ci = entry[1]
                pos = mpos[code.checks[ci][2]]
                bit = (anticommutes(residual, cpaulis[ci])
                       ^ (flips >> ((rounds - 1) * nm + pos))) & 1
            det |= bit << k
        obs = anticommutes(residual, obs_pauli) ^ _parity(flips & fold_mask)
        out.append((det, obs, label))
    return out


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def lc_census(code, layers, site, init_map, se_rounds=1, stim_check=False):
    """Census under the criterion and noise model of Lao-Criger, CF'22
    (DOI 10.1145/3528416.3530237): |Y>-state injection, corrupting =
    escapes their post-selection AND flips M_L ~ Y_L; their circuit-level
    model has NO idle errors, so idle mechanisms are excluded, and the
    remaining locations are reported per parameter class (p2/15 per
    two-qubit Pauli, p1/3 per single-qubit-gate Pauli, p_IN init flips,
    p_M measurement flips) for direct comparison with their Eqs (1)-(3).

    Mapping notes (transcribed from the paper 2026-07-30): their rotated
    lattice coloring is the MIRROR (c -> d-1-c) of this repo's, so their
    corner protocol "CR" (magic qubit D3, Appendix A init) lands on OUR
    top-right corner with init ZZ.ZXXXXX; their "MR" is the center with
    a diagonal corner split (ZXXZ.ZXXZ); their "two rounds of stabiliser
    measurement" = se_rounds=1 here (injection round + one more).  A
    mechanism flips Y_L iff it flips exactly one of Z_L, X_L, so the
    Y-state census is the XOR of the two stand-in frames' observable
    bits (the site's own init flip, whose Pauli differs per frame, is
    judged per frame).  Their per-protocol single-qubit rotation gate
    (the 2/3 p1 term) is NOT a location in our builder -- we initialize
    the stand-in directly -- so p1 counts here exclude it; add their
    2/3 p1 when comparing totals."""
    frames = standard_frames(code, layers, site, init_map, se_rounds)
    report, results, _ = injection_report(code, frames,
                                          stim_check=stim_check)
    byf = {r["frame"]: r for r in results}
    fz, fx = byf["frameZ"], byf["frameX"]
    zmap = {lab: (d, o) for (d, o, lab) in fz["sigs"]}
    xmap = {lab: (d, o) for (d, o, lab) in fx["sigs"]}

    def klass(lab):
        if lab.startswith(("CX(", "CZ(")):
            return "p2_fifteenths"
        if lab.startswith("idle"):
            return "idle"
        if lab.startswith("init"):
            return "p_in"
        if lab.startswith("meas flip"):
            return "p_m"
        return "p1_thirds"

    out = {"p2_fifteenths": 0, "p_in": 0, "p_m": 0, "p1_thirds": 0}
    mechs = []
    mismatched = 0
    for lab, (dz, oz) in zmap.items():
        dx, ox = xmap[lab]
        k = klass(lab)
        if k == "idle":
            continue
        if lab == f"init flip q{site}":
            corrupt = (not dz and oz) or (not dx and ox)
        else:
            if (dz == 0) != (dx == 0):
                mismatched += 1
            corrupt = (not dz) and (not dx) and (oz ^ ox)
        if corrupt:
            out[k] += 1
            mechs.append(lab)
    out["corrupting_mechanisms"] = sorted(mechs)
    out["frame_detect_mismatches"] = mismatched
    if stim_check:
        out["stim_agrees"] = all(f["stim_agrees"]
                                 for f in report["frames"])
    return out


def closed_census(code, layers, rounds, basis):
    """The injection census criterion applied to a CLOSED memory
    experiment (syndrome extraction doing its job): location-level
    counts of accepted-yet-logical-flipping mechanisms and pairs.
    For a schedule with circuit distance d these are zero below weight
    d -- the census is the counting refinement of the distance
    (d_circ = the first weight at which it becomes nonzero)."""
    from spectrum import (closed_signatures, declared_parities,
                          fragment_from_layers)
    fr = fragment_from_layers(code, layers, rounds, basis, basis)
    det_masks, obs_masks = declared_parities(code, rounds, basis)
    sigs, _ = closed_signatures(fr, det_masks, obs_masks)
    nd = len(det_masks)
    detmask = (1 << nd) - 1

    def weight(label):
        if label.startswith(("CX(", "CZ(")):
            return 1 / 15
        if label.startswith(("idle ", "H(", "S(")):
            return 1 / 3
        return 1.0                       # reset / measurement flips

    triples = [(s & detmask, s >> nd, lab)
               for s, (_, _, lab) in zip(sigs, fr.mechs)]
    return census_counts(triples, [weight(lab) for _, _, lab in triples])


def census_counts(sigs, weights):
    """Post-selected corruption census from per-mechanism signatures.
    weight-1: accepted (no detector) and observable-flipping mechanisms.
    weight-2: unordered mechanism pairs whose detector bits cancel and
    whose observable bits differ (subset `both_detected`: both members
    individually fire detectors -- the pairs post-selection can never see
    coming). Weighted numbers are coefficients of p resp. p^2."""
    w1 = [(lab, weights[i]) for i, (d, o, lab) in enumerate(sigs)
          if not d and o]
    # per detector syndrome, tally each observable value (an F2 vector --
    # one bit for injection frames, k bits for closed k>1 memory): a pair
    # is corrupting iff its detector bits cancel and its observable
    # values DIFFER
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for i, (d, o, _) in enumerate(sigs):
        e = buckets[d][o]
        e[0] += 1
        e[1] += weights[i]
    pairs = pairs_bd = 0
    cw = cw_bd = 0.0
    for d, obsmap in buckets.items():
        tot = sum(n for n, _ in obsmap.values())
        wtot = sum(w for _, w in obsmap.values())
        p = (tot * tot - sum(n * n for n, _ in obsmap.values())) // 2
        c = (wtot * wtot - sum(w * w for _, w in obsmap.values())) / 2
        pairs += p
        cw += c
        if d:
            pairs_bd += p
            cw_bd += c
    return {
        "mechanisms": len(sigs),
        "detected": sum(1 for d, _, _ in sigs if d),
        "w1_count": len(w1),
        "w1_weighted": round(sum(w for _, w in w1), 6),
        "w1_labels": sorted(lab for lab, _ in w1),
        "w2_pairs": pairs,
        "w2_pairs_both_detected": pairs_bd,
        "w2_weighted": round(cw, 6),
        "w2_weighted_both_detected": round(cw_bd, 6),
    }


def union_census(frame_results):
    """Location-level union over scoring frames: a mechanism (identified by
    its label) corrupts if it is accepted and observable-flipping in at
    least one frame. Labels present in only one frame (e.g. the stand-in
    rotation's own noise) are judged by that frame alone."""
    corrupt = {}
    for res in frame_results:
        wmap = dict(zip(res["labels"], res["weights"]))
        for d, o, lab in res["sigs"]:
            if not d and o:
                corrupt[lab] = wmap[lab]
    return {"w1_count": len(corrupt),
            "w1_weighted": round(sum(corrupt.values()), 6),
            "w1_labels": sorted(corrupt)}


# ---------------------------------------------------------------------------
# Frame validation against stim (p = 0)
# ---------------------------------------------------------------------------


def _sampled_parities(circ, rec_lists, shots=64, seed=0):
    """Per rec-list: set of sampled parities (a singleton iff deterministic,
    up to 2^-shots false-negative odds; the kept set is certified exactly
    afterwards via stim's detector_error_model)."""
    sample = circ.without_noise().compile_sampler(seed=seed).sample(shots)
    out = []
    for recs in rec_lists:
        col = None
        for i in recs:
            v = sample[:, i]
            col = v if col is None else col ^ v
        out.append(set(bool(x) for x in col))
    return out


def validate_frames(code, frames, candidates, rounds, p=1e-3):
    """Keep the candidates whose parity is deterministic AND identical in
    every frame (including validation-only frames); require each frame's
    observable to be deterministic; certify the kept set by building each
    frame's p=0 detector error model. Returns (kept, dropped)."""
    nm = len(_meas_layout(code)[2])
    nc = len(code.checks)
    rec_lists = [det_recs(code, e, rounds) for e in candidates]
    per_frame = []
    for f in frames:
        circ = experiment_circuit(code, f["round_layers"], f["init_map"],
                                  [], f["obs_pauli"], f["fold_recs"], p)
        obs_recs = [rounds * nm + nc] + sorted(f["fold_recs"])
        ps = _sampled_parities(circ, rec_lists + [obs_recs])
        if len(ps[-1]) != 1:
            raise ValueError(f"{f['name']}: observable not deterministic "
                             "(initialization pattern does not inject)")
        per_frame.append(ps[:-1])
    kept, dropped = [], []
    for k, entry in enumerate(candidates):
        vals = set()
        for ps in per_frame:
            vals |= ps[k]
        (kept if len(vals) == 1 else dropped).append(entry)
    for f in frames:
        circ = experiment_circuit(code, f["round_layers"], f["init_map"],
                                  kept, f["obs_pauli"], f["fold_recs"], 0)
        circ.detector_error_model()          # raises if not deterministic
    return kept, dropped


# ---------------------------------------------------------------------------
# stim cross-check of the weight-1 census (location-level, via the DEM)
# ---------------------------------------------------------------------------


def stim_w1(circ):
    """Number of circuit fault locations that flip the observable and no
    detector, counted through stim's detector error model."""
    dem = circ.detector_error_model(flatten_loops=True)
    filt = stim.DetectorErrorModel()
    for inst in dem:
        if inst.type != "error":
            continue
        targs = inst.targets_copy()
        if (not any(t.is_relative_detector_id() for t in targs)
                and any(t.is_logical_observable_id() for t in targs)):
            filt.append(inst)
    expl = circ.explain_detector_error_model_errors(
        dem_filter=filt, reduce_to_one_representative_error=False)
    return sum(len(e.circuit_error_locations) for e in expl)


# ---------------------------------------------------------------------------
# Schemes (rotated surface code presets; the machinery above is code-generic)
# ---------------------------------------------------------------------------


def surface_pattern(d, r0, c0, style):
    """Init preset for standard injection at data site (r0, c0), Li-style:
    the site's ROW (a Z_L representative through the site) is Z-initialized
    and its COLUMN (an X_L representative) X-initialized, so both frame
    observables are anchored; the remaining bulk follows `style`:
    "diag":  X below the main diagonal through the site, Z above
             (ties: X above the site's row, else Z);
    "col":   X left of the site's column, Z right (ties split by row);
    "row":   Z above the site's row, X below (ties split by column).
    The site itself is left unset -- frames set it."""
    m = {}
    for r in range(d):
        for c in range(d):
            if (r, c) == (r0, c0):
                continue
            if r == r0:
                m[r * d + c] = "Z"
            elif c == c0:
                m[r * d + c] = "X"
            elif style == "diag":
                delta = (r - r0) - (c - c0)
                m[r * d + c] = ("X" if delta > 0 else "Z" if delta < 0
                                else ("X" if r < r0 else "Z"))
            elif style == "col":
                m[r * d + c] = "X" if c < c0 else "Z"
            elif style == "row":
                m[r * d + c] = "Z" if r < r0 else "X"
            else:
                raise ValueError(f"unknown pattern {style}")
    return m


def standard_frames(code, layers, site, init_map, se_rounds):
    """Scoring frames for data-qubit injection: the injected state sits on
    `site`, round 1 is an ordinary SE round (the projection), followed by
    se_rounds more.  Frame observables are REQUIRED: if the frame's
    logical is not realizable as a web, the pattern does not inject."""
    round_layers = [layers] * (1 + se_rounds)
    frames = []
    for fb in ("Z", "X"):
        im = dict(init_map)
        im[site] = fb
        frames.append({"name": f"frame{fb}", "round_layers": round_layers,
                       "init_map": im, "obs_kind": fb, "census": True})
    return frames


def pauli_mul(p1, p2):
    """Product of two Pauli dicts (F2 componentwise; phases irrelevant)."""
    out = dict(p1)
    for q, (x, z) in p2.items():
        ox, oz = out.get(q, (0, 0))
        v = (ox ^ x, oz ^ z)
        if v == (0, 0):
            out.pop(q, None)
        else:
            out[q] = v
    return out


def zcheck_hook_frames(code, check_idx, tail, lz, lx, layers, se_rounds,
                       init_map=None, r0_checks="inj"):
    """Generic hook injection on a Z-type check of ANY CSS code.

    Round 1 measures ONLY this check (head CNOTs, then the injected
    rotation exp(-i theta Z/2) on its own ancilla, then the tail CNOTs);
    the rotation axis is Z on `tail`.  The caller supplies the algebra
    that makes this an injection (see zcheck_hook_config, which derives it):
      lz  a Z-logical representative with tail <= supp(lz); the
          remainder R = supp(lz) - tail must be Z-initialized, so the
          axis equals that logical on the post-selected code;
      lx  an X-logical representative anticommuting with lz, disjoint
          from R, with even overlap with the check support (so it
          survives the round-1 projection); X-initialized => the
          unrotated logical state is |+~>.
    Rounds 2+ grow/stabilize with the given full SE layers.  Frames:
    theta = 0 (obs = lx), 90 deg = S (obs = lx*lz, the magic plane),
    180 deg = S,S (validation-only).  Default init: Z everywhere except
    X on supp(lx).

    r0_checks: which checks the injection round measures besides the
    hook check itself --
      "inj"   none (minimal exposure; on expander codes this leaves the
              injected logical's support unwatched for a full round);
      "xsafe" every X-check disjoint from the tail (all of which commute
              with the mid-round rotation axis Z(tail)), run with the
              growth schedule's layer assignment BEFORE the hook
              sequence: their first-round outcomes are projective but
              make the round-1 X-compares declarable -- the checks that
              actually watch the injected X-logical -- without paying
              the fault locations of round-0 Z-check CNOTs;
      "safe"  "xsafe" plus every other Z-check, X-restricted layers
              first (the ordering is free in the location model and
              maximizes what the round-1 compares can see).
    Everything is re-derived and validated downstream (injection_report):
    an unsound configuration fails loudly."""
    kind, sup, anc = code.checks[check_idx]
    if kind != "Z":
        raise ValueError("zcheck_hook_frames needs a Z-type check")
    tail = tuple(tail)
    if not set(tail) <= set(sup):
        raise ValueError("tail must be a subset of the check support")
    head = tuple(sorted(set(sup) - set(tail)))
    if init_map is None:
        init_map = {q: "Z" for q in range(code.n)}
        for q in lx:
            init_map[q] = "X"
    obs_x = {q: (1, 0) for q in lx}
    obs_y = pauli_mul(obs_x, {q: (0, 1) for q in lz})

    pre = []
    if r0_checks in ("safe", "xsafe"):
        xa = {a for k, s, a in code.checks
              if k == "X" and not (set(s) & set(tail))}
        za = {a for k, _, a in code.checks
              if k == "Z" and a != anc} if r0_checks == "safe" else set()
        for ancs in (xa, za):
            for layer in layers:
                keep = [(n, qs) for n, qs in layer
                        if n == "CX" and (set(qs) & ancs)]
                if keep:
                    pre.append(keep)
    elif r0_checks != "inj":
        raise ValueError(f"unknown r0_checks mode {r0_checks!r}")

    def rl(insert):
        r1 = [list(g) for g in pre]
        r1 += [[("CX", (q, anc))] for q in head]
        for g in insert:
            r1.append([(g, (anc,))])
        r1 += [[("CX", (q, anc))] for q in tail]
        return [r1] + [list(layers)] * se_rounds

    frames = [
        {"name": "frameX", "round_layers": rl(()),
         "init_map": dict(init_map), "obs_kind": "X", "obs_pauli": obs_x,
         "census": True},
        {"name": "frameY", "round_layers": rl(("S",)),
         "init_map": dict(init_map), "obs_kind": "Y", "obs_pauli": obs_y,
         "census": True},
        {"name": "frameZrot", "round_layers": rl(("S", "S")),
         "init_map": dict(init_map), "obs_kind": "X", "obs_pauli": obs_x,
         "census": False},
    ]
    return frames


def zcheck_hook_config(code, lz, check_idx=None, tail_size=None, lx_seed=0):
    """Derive a hook-injection configuration for a CSS code from a
    Z-logical representative `lz` (e.g. from
    codes.low_weight_logical_reps):

      check: a Z-check overlapping supp(lz) (the overlap is the tail --
             the hook CNOTs fired after the rotation);
      lx:    an X-logical partner solved over ker(Hz) with
             supp(lx) disjoint from R = supp(lz)-tail,
             |lx & tail| odd (anticommutes with the injected logical;
             also proves lx's class nontrivial), and
             |lx & supp(check)| even (survives the round-1 projection);
             weight-minimized by randomized descent over the solution
             space.

    Returns (check_idx, tail, lx) or raises if no configuration exists."""
    from f2 import nullspace, rref

    lzset = set(lz)
    options = []
    for ci, (kind, sup, _) in enumerate(code.checks):
        if kind != "Z":
            continue
        ov = tuple(sorted(lzset & set(sup)))
        if not ov or len(ov) == len(sup):
            continue
        if check_idx is not None and ci != check_idx:
            continue
        if tail_size is not None and len(ov) != tail_size:
            continue
        options.append((ci, ov))
    if not options:
        raise ValueError("no Z-check with a compatible overlap on this "
                         "logical representative")

    hz = []
    for kind, sup, _ in code.checks:
        if kind == "Z":
            m = 0
            for q in sup:
                m |= 1 << q
            hz.append(m)
    xnorm = nullspace(hz, code.n)          # X-type normalizer

    def mask(qs):
        m = 0
        for q in qs:
            m |= 1 << q
        return m

    for ci, tail in options:
        sup = code.checks[ci][1]
        R = lzset - set(tail)
        tmask, smask = mask(tail), mask(sup)
        # constraints on v in span(xnorm): v_q = 0 (q in R),
        # parity(v & tmask) = 1, parity(v & smask) = 0
        cons = [(1 << q, 0) for q in sorted(R)]
        cons += [(tmask, 1), (smask, 0)]
        piv = {}
        for j, b in enumerate(xnorm):
            row = 0
            for k, (cm, _) in enumerate(cons):
                if bin(b & cm).count("1") & 1:
                    row |= 1 << k
            row |= 1 << (len(cons) + j)     # track the combination
            cmask = (1 << len(cons)) - 1
            r = row
            while r & cmask:
                c = (r & cmask).bit_length() - 1
                if c in piv:
                    r ^= piv[c]
                else:
                    piv[c] = r
                    break
        target = 0
        for k, (_, val) in enumerate(cons):
            if val:
                target |= 1 << k
        r = target
        ok = True
        while r & ((1 << len(cons)) - 1):
            c = (r & ((1 << len(cons)) - 1)).bit_length() - 1
            if c not in piv:
                ok = False
                break
            r ^= piv[c]
        if not ok:
            continue
        v = 0
        for j, b in enumerate(xnorm):
            if (r >> (len(cons) + j)) & 1:
                v ^= b
        # weight-lowering descent over the homogeneous solutions
        # (lx_seed varies the walk: distinct same-weight partners differ
        # in geometry and therefore in census)
        hom = [h for h in _hom_solutions(xnorm, cons)]
        import random
        rng = random.Random(lx_seed)
        for _ in range(3000):
            if not hom:
                break
            cand = v ^ rng.choice(hom)
            if bin(cand).count("1") <= bin(v).count("1") and cand != v:
                if (bin(cand).count("1") < bin(v).count("1")
                        or rng.random() < 0.3):
                    v = cand
        lx = tuple(q for q in range(code.n) if (v >> q) & 1)
        return ci, tail, lx
    raise ValueError("no X-partner representative satisfies the "
                     "hook constraints for any candidate check")


def _hom_solutions(basis, cons):
    """Basis of homogeneous solutions: combinations of `basis` vectors
    satisfying all constraints with parity 0."""
    rows = []
    for b in basis:
        row = 0
        for k, (cm, _) in enumerate(cons):
            if bin(b & cm).count("1") & 1:
                row |= 1 << k
        rows.append(row)
    from f2 import nullspace as ns
    combos = ns([sum(((rows[j] >> k) & 1) << j for j in range(len(rows)))
                 for k in range(len(cons))], len(basis))
    out = []
    for cmb in combos:
        v = 0
        for j, b in enumerate(basis):
            if (cmb >> j) & 1:
                v ^= b
        if v:
            out.append(v)
    return out


def hook_init_pattern(code, lx, R, hard=()):
    """Init bipartition for a generic (e.g. qLDPC) hook injection,
    optimized with z3 MaxSAT: qubit -> "X"/"Z" with supp(lx) forced X and
    R (the Z-logical remainder) forced Z, maximizing the number of
    checks whose support is pure in their own basis -- each such check is
    a declarable first-full-round detector, closing the idle/init blind
    spots around the injected logical (the qLDPC analog of the surface
    code's diagonal init split, where geometry no longer provides one)."""
    import z3
    opt = z3.Optimize()
    v = [z3.Bool(f"x{q}") for q in range(code.n)]
    for q in lx:
        opt.add(v[q])
    for q in R:
        opt.add(z3.Not(v[q]))
    for q, b in hard:
        opt.add(v[q] if b == "X" else z3.Not(v[q]))
    for kind, sup, _ in code.checks:
        lits = [v[q] if kind == "X" else z3.Not(v[q]) for q in sup]
        opt.add_soft(z3.And(lits))
    if opt.check() != z3.sat:
        raise ValueError("no init pattern satisfies the hard constraints")
    m = opt.model()
    return {q: "X" if z3.is_true(m[v[q]]) else "Z"
            for q in range(code.n)}


def hook_frames(code, d, layers, se_rounds, anchor=0, init_map=None,
                head_first=True, tail=None):
    """Scoring + validation frames for hook injection (Gidney
    arXiv:2302.12292, transcribed to this project's builders).

    Round 1 measures ONLY the four-body X stabilizer of the 2x2 corner
    subpatch at `anchor` (its data block hosts a full-patch X-check whose
    own ancilla is used), with the injected rotation exp(-i theta X/2)
    inserted on the ancilla between its 2nd and 3rd CNOT.  The last two
    CNOTs target one COLUMN of the block, so the rotation's hook is
    exactly that column pair = the subpatch's logical X -- the deliberate
    hook that rotates the logical.  All subpatch qubits start |0> (the
    two-body Z checks of the distance-2 patch are init-definite, so, per
    Gidney's no-op removal, they are not measured in round 1); the
    unrotated logical state is |0~>, and the hook sweeps the logical
    ZY plane.  Rounds 2+ grow the subpatch into the full patch by
    ordinary syndrome extraction; post-selection detectors and the
    teleportation corrections (ancilla records folded into the
    observable) are derived mechanically by injection_report.

    Clifford stand-ins for theta: 0 -> "frameZ" (observable Z_L),
    90 deg = sqrt(X) = H S H -> "frameY" (observable Y_L, the magic
    plane), 180 deg = X = H S S H -> "frameZrot" (VALIDATION-ONLY; a
    record parity's expectation is A + B cos(theta) + C sin(theta), so
    constancy at all three angles proves determinism for every injected
    angle, including the non-Clifford T-analog).

    Default init: the diagonal split through the subpatch (Gidney's
    "initialization basis switches across the patch's diagonal"): |+> on
    qubits strictly below the block and strictly below the diagonal
    (r - anchor_row > c - anchor_col), |0> elsewhere -- this anchors the
    grown Z_L along the block's row and transports X_L down the block's
    column, and reduces to the d=3 pattern that every pinned number uses.
    head_first orders the non-hook column before the hook column.
    Returns (frames, hook_pair)."""
    r, c = divmod(anchor, d)
    if r >= d - 1 or c >= d - 1:
        raise ValueError(f"anchor {anchor} has no 2x2 subpatch")
    block = {anchor, anchor + 1, anchor + d, anchor + d + 1}
    ci = next((i for i, (k, s, _) in enumerate(code.checks)
               if k == "X" and set(s) == block), None)
    if ci is None:
        raise ValueError(f"no four-body X-check on block {sorted(block)}; "
                         "choose an anchor whose 2x2 cell is X-type")
    anc = code.checks[ci][2]
    if tail is None:
        tail = (anchor, anchor + d)                # hook column = X_L~
        head = (anchor + 1, anchor + d + 1)
        if not head_first:
            head, tail = tail, head
    else:
        # explicit split (position/orientation sweeps): non-logical tails
        # are expected to be REJECTED downstream ("does not inject")
        tail = tuple(tail)
        head = tuple(sorted(block - set(tail)))

    if init_map is None:
        init_map = {}
        for qr in range(d):
            for qc in range(d):
                below_diag = (qr - r) > (qc - c)
                init_map[qr * d + qc] = ("X" if below_diag and qr > r + 1
                                         else "Z")

    def rl(insert):
        r1 = [[("CX", (anc, q))] for q in head]
        for g in insert:
            r1.append([(g, (anc,))])
        r1 += [[("CX", (anc, q))] for q in tail]
        return [r1] + [list(layers)] * se_rounds

    frames = [
        {"name": "frameZ", "round_layers": rl(()),
         "init_map": dict(init_map), "obs_kind": "Z", "census": True},
        {"name": "frameY", "round_layers": rl(("H", "S", "H")),
         "init_map": dict(init_map), "obs_kind": "Y", "census": True},
        {"name": "frameZrot", "round_layers": rl(("H", "S", "S", "H")),
         "init_map": dict(init_map), "obs_kind": "Z", "census": False},
    ]
    return frames, tail


# ---------------------------------------------------------------------------
# The full pipeline
# ---------------------------------------------------------------------------


def injection_report(code, frames, p=1e-3, stim_check=True):
    """Validate declarations across all frames, run the census on each
    scoring frame through the webs engine, cross-check weight-1 counts
    against stim's DEM, and take the location-level union."""
    rounds = {len(f["round_layers"]) for f in frames}
    assert len(rounds) == 1
    rounds = rounds.pop()

    # per frame: fragment + the frame's REQUIRED observable as a web
    # (obs_pauli may be given explicitly -- e.g. a specific logical qubit
    # of a k>1 code -- else it is code.logical_{x,z,y}[0] via obs_kind)
    for f in frames:
        fr, weights = experiment_fragment(code, f["round_layers"],
                                          f["init_map"])
        f["fragment"], f["weights"] = fr, weights
        if "obs_pauli" not in f:
            f["obs_pauli"] = logical_pauli(code, f["obs_kind"])
        mask = observable_fold(fr, code, f["obs_pauli"])
        if mask is None:
            raise ValueError(
                f"{f['name']}: {f['obs_kind']}_L is not realizable as a "
                "web -- this configuration does not inject")
        f["fold_mask"] = mask
        f["fold_recs"] = [i for i in range(len(fr.meas))
                          if (mask >> i) & 1]

    candidates = candidate_detectors(code, rounds)
    kept, dropped = validate_frames(code, frames, candidates, rounds, p)

    results = []
    for f in frames:
        if not f["census"]:
            continue
        sigs = frame_signatures(f["fragment"], code, kept, f["obs_pauli"],
                                f["fold_mask"], rounds)
        row = {"frame": f["name"], "observable": f["obs_kind"] + "_L",
               "folded_records": len(f["fold_recs"]),
               "sigs": sigs, "labels": [m[2] for m in f["fragment"].mechs],
               "weights": f["weights"]}
        row.update(census_counts(sigs, f["weights"]))
        if stim_check:
            circ = experiment_circuit(code, f["round_layers"], f["init_map"],
                                      kept, f["obs_pauli"], f["fold_recs"], p)
            row["stim_w1"] = stim_w1(circ)
            row["stim_agrees"] = row["stim_w1"] == row["w1_count"]
        results.append(row)

    report = {
        "rounds": rounds,
        "detectors_kept": len(kept),
        "detectors_dropped": [list(e) for e in dropped],
        "frames": [{k: v for k, v in r.items()
                    if k not in ("sigs", "labels", "weights")}
                   for r in results],
        "union": union_census(results),
    }
    return report, results, kept


def frame_circuits(code, frames, kept, p=1e-3):
    """[(frame name, stim circuit)] with the validated declarations --
    exactly what the census scored (for visualization / export)."""
    return [(f["name"],
             experiment_circuit(code, f["round_layers"], f["init_map"],
                                kept, f["obs_pauli"], f["fold_recs"], p))
            for f in frames]
