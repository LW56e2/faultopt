"""TOOL: the Pauli-web engine — the seed of the open-fragment grader.

A web is a consistent Pauli labeling of a circuit fragment's wire segments
(spider-stabilizer condition at every op). All conditions are F2-linear, so:
  webs                = nullspace of a sparse binary constraint matrix
  internal detectors  = webs acting trivially on the open boundary
  fault syndrome      = anticommutation (symplectic form) with detector webs
  fault effect class  = anticommutation with the full web basis
                        (spacetime equivalence)

This generalizes stim's detectors to OPEN fragments (unreset inputs,
unmeasured outputs), which stim cannot represent. A separate forward
Pauli-propagation engine provides (measurement flips, boundary residual)
per fault and cross-validates the web-based classification.

Constraint dictionary (per op, on segment variables x_s, z_s):
  gate:   output Pauli = symplectic action on input Pauli
  R  (Z): x_out = 0            RX:  z_out = 0
  M  (Z): x_in = 0, measurement included in web parity iff z_in = 1
  MR (Z): x_in = 0, x_out = 0, inclusion = z_in     (MRX: swap x<->z)
  open boundary segments: unconstrained.
"""

import itertools

from codes import layer_gates
from f2 import in_span, nullspace, rank, rref  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# Circuit fragments: ops acting on wire segments
# ---------------------------------------------------------------------------


class Fragment:
    def __init__(self):
        self.ops = []            # (name, qubits) in time order
        self.nseg = 0
        self.cur = {}            # qubit -> current segment
        self.rows = []           # F2 constraint rows over segment vars
        self.boundary_in = []    # open input segments
        self.meas = []           # (inclusion_var, label) per measurement
        self.mechs = []          # (op_idx, {qubit: (x,z)}, label)
        self.terminated = set()

    # -- variable helpers ---------------------------------------------------
    def _x(self, s):
        return 2 * s

    def _z(self, s):
        return 2 * s + 1

    def _new_seg(self):
        self.nseg += 1
        return self.nseg - 1

    def _get(self, q):
        """Current segment of q; entering the fragment openly if untouched."""
        if q not in self.cur:
            s = self._new_seg()
            self.cur[q] = s
            self.boundary_in.append(s)
        return self.cur[q]

    # -- ops ------------------------------------------------------------
    def reset(self, q, basis):
        s = self._new_seg()
        self.cur[q] = s
        v = self._x(s) if basis == "Z" else self._z(s)
        self.rows.append(1 << v)
        self.ops.append(("R" + basis, (q,)))

    def cx(self, c, t):
        sc, st = self._get(c), self._get(t)
        nc, nt = self._new_seg(), self._new_seg()
        X, Z = self._x, self._z
        # x_c' = x_c ; z_c' = z_c + z_t ; x_t' = x_t + x_c ; z_t' = z_t
        self.rows += [
            (1 << X(nc)) | (1 << X(sc)),
            (1 << Z(nc)) | (1 << Z(sc)) | (1 << Z(st)),
            (1 << X(nt)) | (1 << X(st)) | (1 << X(sc)),
            (1 << Z(nt)) | (1 << Z(st)),
        ]
        self.cur[c], self.cur[t] = nc, nt
        self.ops.append(("CX", (c, t)))

    def cz(self, a, b):
        sa, sb = self._get(a), self._get(b)
        na, nb = self._new_seg(), self._new_seg()
        X, Z = self._x, self._z
        # x' = x ; z_a' = z_a + x_b ; z_b' = z_b + x_a
        self.rows += [
            (1 << X(na)) | (1 << X(sa)),
            (1 << Z(na)) | (1 << Z(sa)) | (1 << X(sb)),
            (1 << X(nb)) | (1 << X(sb)),
            (1 << Z(nb)) | (1 << Z(sb)) | (1 << X(sa)),
        ]
        self.cur[a], self.cur[b] = na, nb
        self.ops.append(("CZ", (a, b)))

    def h(self, q):
        s = self._get(q)
        n = self._new_seg()
        X, Z = self._x, self._z
        # x' = z ; z' = x
        self.rows += [(1 << X(n)) | (1 << Z(s)),
                      (1 << Z(n)) | (1 << X(s))]
        self.cur[q] = n
        self.ops.append(("H", (q,)))

    def s_gate(self, q):
        s = self._get(q)
        n = self._new_seg()
        X, Z = self._x, self._z
        # x' = x ; z' = z + x
        self.rows += [(1 << X(n)) | (1 << X(s)),
                      (1 << Z(n)) | (1 << Z(s)) | (1 << X(s))]
        self.cur[q] = n
        self.ops.append(("S", (q,)))

    def measure(self, q, basis, label, reset=False):
        s = self._get(q)
        dead = self._x(s) if basis == "Z" else self._z(s)
        incl = self._z(s) if basis == "Z" else self._x(s)
        self.rows.append(1 << dead)
        self.meas.append((incl, label))
        if reset:
            ns = self._new_seg()
            self.cur[q] = ns
            v = self._x(ns) if basis == "Z" else self._z(ns)
            self.rows.append(1 << v)
        else:
            self.terminated.add(q)
        self.ops.append(("M" + basis + ("R" if reset else ""), (q,)))

    def add_mech(self, paulis, label):
        """Fault mechanism: Pauli on the CURRENT segments of given qubits,
        occurring after the last appended op."""
        self.mechs.append((len(self.ops) - 1, dict(paulis), label))

    # -- analysis ---------------------------------------------------------
    def boundary_out(self):
        return [s for q, s in self.cur.items() if q not in self.terminated]

    def web_basis(self):
        return nullspace(self.rows, 2 * self.nseg)

    def detector_basis(self, webs=None):
        """Sub-basis of webs with trivial action on open boundary segments,
        projected to measurement-inclusion masks (the detectors).

        Ancilla wires ending in a fresh reset are treated conservatively as
        boundary (a web ending in the reset basis there is physically valid
        but redundant)."""
        webs = webs or self.web_basis()
        bnd_vars = set()
        for s in self.boundary_in + self.boundary_out():
            bnd_vars |= {self._x(s), self._z(s)}
        piv = {}
        trivial = []
        for w in webs:
            r = w
            changed = True
            while changed:
                changed = False
                for v in list(piv):
                    if (r >> v) & 1:
                        r ^= piv[v]
                        changed = True
            hit = [v for v in bnd_vars if (r >> v) & 1]
            if hit:
                piv[max(hit)] = r
            elif r:
                trivial.append(r)
        dets = []
        for w in trivial:
            mask = 0
            for i, (var, _) in enumerate(self.meas):
                if (w >> var) & 1:
                    mask |= 1 << i
            if mask:
                dets.append((w, mask))
        return trivial, dets

    # -- Pauli propagation (independent engine, used for cross-validation) --
    def propagate(self, op_idx, paulis):
        """Push Pauli {q:(x,z)} inserted after op op_idx to the end.
        Returns (measurement flip mask, residual {q:(x,z)})."""
        pauli = {q: p for q, p in paulis.items() if p != (0, 0)}
        flips = 0
        mcount = sum(1 for i in range(op_idx + 1)
                     if self.ops[i][0].startswith("M"))
        for i in range(op_idx + 1, len(self.ops)):
            name, qs = self.ops[i]
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
                anti = x if name[1] == "Z" else z
                if anti:
                    flips |= 1 << mcount
                mcount += 1
                pauli.pop(q, None)
        residual = {q: p for q, p in pauli.items()
                    if p != (0, 0) and q not in self.terminated}
        return flips, residual

    def mech_signature(self, dets):
        """For each mechanism: (detector flip mask, flips, residual, label)."""
        out = []
        for op_idx, paulis, label in self.mechs:
            flips, residual = self.propagate(op_idx, paulis)
            dmask = 0
            for j, (_, incl) in enumerate(dets):
                if bin(flips & incl).count("1") & 1:
                    dmask |= 1 << j
            out.append((dmask, flips, residual, label))
        return out


# ---------------------------------------------------------------------------
# Syndrome-extraction fragments (mirrors the scoring.py circuits)
# ---------------------------------------------------------------------------

PAULIS2 = [(x1, z1, x2, z2)
           for x1, z1, x2, z2 in itertools.product((0, 1), repeat=4)
           if (x1, z1, x2, z2) != (0, 0, 0, 0)]

_P = {(1, 0): "X", (0, 1): "Z", (1, 1): "Y"}


def se_fragment(d, checks, sched_z, sched_x, rounds, closed_basis=None,
                cx_noise=True, other_noise=True):
    """Syndrome-extraction fragment under uniform schedules.

    closed_basis: None -> open data legs; "Z"/"X" -> closed memory expt."""
    fr = Fragment()
    data = list(range(d * d))
    zanc = [ch.anc for ch in checks if ch.kind == "Z"]
    xanc = [ch.anc for ch in checks if ch.kind == "X"]

    if closed_basis:
        for q in data:
            fr.reset(q, closed_basis)
            if other_noise:
                fr.add_mech({q: (1, 0) if closed_basis == "Z" else (0, 1)},
                            f"init flip q{q}")
    for q in zanc:
        fr.reset(q, "Z")
        if other_noise:
            fr.add_mech({q: (1, 0)}, f"init flip anc{q}")
    for q in xanc:
        fr.reset(q, "X")
        if other_noise:
            fr.add_mech({q: (0, 1)}, f"init flip anc{q}")

    for t in range(rounds):
        if other_noise and fr.ops:
            for q in data:
                fr._get(q)
            for q in data:
                for p in _P:
                    fr.add_mech({q: p}, f"idle {_P[p]} q{q} r{t}")
        for layer in range(4):
            for ctrl, targ in layer_gates(checks, sched_z, sched_x, layer):
                fr.cx(ctrl, targ)
                if cx_noise:
                    for x1, z1, x2, z2 in PAULIS2:
                        fr.add_mech(
                            {ctrl: (x1, z1), targ: (x2, z2)},
                            f"CX({ctrl}->{targ}) r{t} L{layer+1} "
                            f"{_P.get((x1, z1), 'I')}{_P.get((x2, z2), 'I')}")
        for q in zanc:
            if other_noise:
                fr._get(q)
                fr.add_mech({q: (1, 0)}, f"meas flip anc{q} r{t}")
            fr.measure(q, "Z", f"anc{q}:r{t}", reset=True)
        for q in xanc:
            if other_noise:
                fr._get(q)
                fr.add_mech({q: (0, 1)}, f"meas flip anc{q} r{t}")
            fr.measure(q, "X", f"anc{q}:r{t}", reset=True)

    if closed_basis:
        for q in data:
            if other_noise:
                fr.add_mech({q: (1, 0) if closed_basis == "Z" else (0, 1)},
                            f"final meas flip q{q}")
            fr.measure(q, closed_basis, f"data{q}", reset=False)
    return fr


def declared_detectors_and_obs(d, checks, rounds, basis="Z"):
    """The stim-style declared detector parities + observable, as inclusion
    masks over the fragment's measurement list (ancillas per round, then
    data). Used to validate the web-derived space against ground truth."""
    zanc = [ch.anc for ch in checks if ch.kind == "Z"]
    xanc = [ch.anc for ch in checks if ch.kind == "X"]
    per_round = zanc + xanc
    idx = {}
    for t in range(rounds):
        for i, q in enumerate(per_round):
            idx[f"anc{q}:r{t}"] = t * len(per_round) + i
    base = rounds * len(per_round)
    for q in range(d * d):
        idx[f"data{q}"] = base + q

    masks = []
    first = zanc if basis == "Z" else xanc
    for q in first:
        masks.append(1 << idx[f"anc{q}:r0"])
    for t in range(1, rounds):
        for q in per_round:
            masks.append((1 << idx[f"anc{q}:r{t}"]) |
                         (1 << idx[f"anc{q}:r{t-1}"]))
    for ch in checks:
        if ch.kind != basis:
            continue
        m = 1 << idx[f"anc{ch.anc}:r{rounds-1}"]
        for _, q in ch.data:
            m |= 1 << idx[f"data{q}"]
        masks.append(m)
    if basis == "Z":
        obs = 0
        for c in range(d):
            obs |= 1 << idx[f"data{c}"]          # top row
    else:
        obs = 0
        for r in range(d):
            obs |= 1 << idx[f"data{r * d}"]      # left column
    return masks, obs


def orientation(residual, d):
    """Orientation of a weight-2 X residual on the d x d data grid."""
    sup = sorted(q for q, (x, z) in residual.items() if x)
    if len(sup) != 2:
        return None
    (r1, c1), (r2, c2) = [(q // d, q % d) for q in sup]
    if c1 == c2:
        return "VERTICAL"
    if r1 == r2:
        return "horizontal"
    return "diagonal"
