"""TOOL: constraint-based candidate generation (the box-as-SAT encoding).

The pieces, separated so each concept from the design is one object:

  GATE_RULES   how each Clifford gate updates tracked-Pauli bits
               (conditional-XOR form; Clifford conjugation is F2-linear).
  Gate/library the MOVE SET: which gates are allowed to fire.
               scheduling_library = one fixed-direction CNOT per Tanner edge
               (the classic scheduling problem); clifford_library = both CNOT
               directions + CZ per edge, H/S on ancillas (general search).
  PauliFlow    one semantic requirement: a tracked Pauli with prescribed
               start, prescribed end on data, and prescribed measurement-flip
               bits on ancillas (the commuting bit is free -- realization-
               independent).
  se_spec      the syndrome-extraction spec as flows: each check operator is
               read out flip-free, and every data Pauli is preserved with
               exactly the physically required outcome correlations.
  Synthesizer  variables b[gate, layer] + structure + flows; one check() =
               one candidate; enumerate() = solve, add blocking clause,
               repeat.
  find_min_depth  the outer budget loop: UNSAT at T means *proof* that no
               circuit of depth T exists; first SAT = depth-optimal.

Non-Clifford gates cannot appear here: the entire encoding is Pauli
propagation, which T-gates break. Non-unitary events (reset/measure) live in
the fixed skeleton; flows see them as boundary conditions.
"""

import itertools
from dataclasses import dataclass, field

import z3


# ---------------------------------------------------------------------------
# Gates and libraries (the move set)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    name: str            # "CX", "CZ", "H", "S"
    qubits: tuple        # CX: (control, target); CZ: sorted pair; 1q: (q,)
    edge: tuple = None   # (check_idx, data_qubit) when realizing a Tanner edge


def scheduling_library(code):
    """v1 move set: one CNOT per Tanner edge, textbook direction
    (Z-check: data->ancilla, X-check: ancilla->data), each fired exactly
    once. Searching = choosing the layer of each edge."""
    gates = []
    for i, (kind, sup, anc) in enumerate(code.checks):
        for q in sup:
            cx = (q, anc) if kind == "Z" else (anc, q)
            gates.append(Gate("CX", cx, edge=(i, q)))
    return gates


def clifford_library(code, one_qubit=("H", "S"), ancilla_only_1q=True):
    """General move set: per Tanner edge, CNOT in BOTH directions and CZ;
    plus 1-qubit Cliffords (default: on ancillas). No exactly-once
    constraint -- gates may fire in any multiplicity the flows allow.
    Two-qubit gates stay on Tanner edges (the code's connectivity)."""
    gates, seen = [], set()
    for i, (kind, sup, anc) in enumerate(code.checks):
        for q in sup:
            for g in (Gate("CX", (q, anc)), Gate("CX", (anc, q)),
                      Gate("CZ", tuple(sorted((q, anc))))):
                if (g.name, g.qubits) not in seen:
                    seen.add((g.name, g.qubits))
                    gates.append(g)
    ancs = [a for _, _, a in code.checks]
    targets = ancs if ancilla_only_1q else list(range(code.n)) + ancs
    for q in targets:
        for name in one_qubit:
            gates.append(Gate(name, (q,)))
    return gates


# ---------------------------------------------------------------------------
# Semantics: Pauli flows (the spec)
# ---------------------------------------------------------------------------


@dataclass
class PauliFlow:
    """Tracked Pauli: one semantic requirement.

    Start: by default fixed to (init_x, init_z). Per-qubit overrides via
    init_mode[q]:
      "Z"    -- qubit starts reset to |0>: start Pauli in {I, Z}
                (x-bit forced 0, z-bit a free solver choice)
      "X"    -- reset to |+>: z-bit forced 0, x-bit free
      "beta" -- reset basis itself is a solver choice (shared selector
                beta_q: True -> |0>, False -> |+>)
      "free" -- open leg, unconstrained
    End: fixed to (fin_x, fin_z) on data qubits; on each measured ancilla
    only the measurement-ANTIcommuting bit is prescribed by flip[anc]
    (0/1), or left free with flip[anc] = None. The commuting bit is always
    free -- it cannot affect any outcome, and fixing it would exclude
    valid realizations."""
    label: str
    init_x: frozenset
    init_z: frozenset
    fin_x: frozenset
    fin_z: frozenset
    flip: dict                    # ancilla -> 0 | 1 | None (free)
    init_mode: dict = field(default_factory=dict)   # qubit -> mode


def se_spec(code, data_flows=True):
    """Syndrome-extraction spec for one round.

    Flows:
      per check i:  S_i (x) B_anc  ->  flip-free everywhere, identity on data
                    ("some ancilla outcome reads S_i, nothing is corrupted")
      per data Pauli P_q:  P_q -> P_q on data, flipping exactly the outcomes
                    of checks that anticommute with P_q (required physics).

    data_flows=False drops the per-data-qubit flows. This is sound for the
    scheduling library (CNOT-only, each edge exactly once): data Paulis then
    propagate only by even-overlap kicks that cancel pairwise, so
    preservation is automatic -- and it shrinks large instances (2n fewer
    flows). Keep True for richer libraries.

    Returns (flows, meas_basis) with meas_basis[anc] = "Z" or "X"."""
    meas_basis = {anc: kind for kind, _, anc in code.checks}
    flows = []
    for i, (kind, sup, anc) in enumerate(code.checks):
        s = frozenset(sup) | {anc}
        flows.append(PauliFlow(
            f"{kind}-check {i} readout",
            init_x=s if kind == "X" else frozenset(),
            init_z=s if kind == "Z" else frozenset(),
            fin_x=frozenset(), fin_z=frozenset(),
            flip={a: 0 for a in meas_basis}))
    if data_flows:
        for q in range(code.n):
            for P in "XZ":
                flip = {anc: int(q in sup and kind != P)
                        for kind, sup, anc in code.checks}
                one = frozenset([q])
                flows.append(PauliFlow(
                    f"{P}{q} preserved",
                    init_x=one if P == "X" else frozenset(),
                    init_z=one if P == "Z" else frozenset(),
                    fin_x=one if P == "X" else frozenset(),
                    fin_z=one if P == "Z" else frozenset(),
                    flip=flip))
    return flows, meas_basis


# ---------------------------------------------------------------------------
# The synthesizer
# ---------------------------------------------------------------------------


class Synthesizer:
    """SAT instance: selectors b[gate, layer] + structure + flow semantics.

    Structural constraints:
      * <=1 gate touching each qubit per layer (parallelism);
      * scheduling mode only: each Tanner-edge CNOT fires exactly once;
      * optional max_gates budget (anti-junk knob for rich libraries).
    Semantic constraints: every PauliFlow, propagated by conditional XOR
    (see GATE update rules inline), meets its endpoint."""

    def __init__(self, code, T, library, flows, meas_basis,
                 exactly_once_edges=False, max_gates=None, tie=None,
                 free_init_basis=()):
        """tie: optional gate -> hashable class. Gates in the same class
        share one selector per layer (they fire together) -- orbit-tying
        for symmetric codes (e.g. BB translation classes), collapsing the
        search space the way uniform schedules do for the surface code."""
        self.code, self.T, self.library = code, T, library
        nq = code.n + len(code.checks)
        self.solver = z3.Solver()
        # per-qubit init-basis selectors (True -> |0>, False -> |+>) for
        # qubits whose reset basis the solver may choose ("beta" init mode)
        self.beta = {q: z3.Bool(f"beta_{q}") for q in free_init_basis}
        self.gate_key = [tie(g) if tie and tie(g) is not None else ("g", gi)
                         for gi, g in enumerate(library)]
        keys = sorted(set(self.gate_key), key=repr)
        self.b = {(k, t): z3.Bool(f"k{ki}_t{t}")
                  for ki, k in enumerate(keys) for t in range(T)}
        self.gate_sel = {(gi, t): self.b[self.gate_key[gi], t]
                         for gi in range(len(library)) for t in range(T)}
        # tied gates must never share a qubit (they always fire together)
        qubits_of = {}
        for gi, g in enumerate(library):
            seen = qubits_of.setdefault(self.gate_key[gi], set())
            if seen & set(g.qubits):
                raise ValueError(f"tie class {self.gate_key[gi]} has "
                                 f"gates sharing a qubit")
            seen |= set(g.qubits)
        # edge CX selectors, for hook clauses / assumption-based validation
        self.edge_sel = {(g.edge, t): self.gate_sel[gi, t]
                         for gi, g in enumerate(library) if g.edge
                         for t in range(T)}

        # structure: parallelism (dedupe by key: tied gates on one qubit
        # were excluded above, so each literal appears at most once)
        touch = {}
        for gi, g in enumerate(library):
            for q in g.qubits:
                for t in range(T):
                    touch.setdefault((q, t), {})[self.gate_key[gi]] = \
                        self.gate_sel[gi, t]
        for lits in touch.values():
            if len(lits) > 1:
                self.solver.add(z3.PbLe(
                    [(l, 1) for l in lits.values()], 1))
        # structure: scheduling mode fires each edge (class) exactly once
        if exactly_once_edges:
            for k in {self.gate_key[gi]
                      for gi, g in enumerate(library) if g.edge}:
                self.solver.add(z3.PbEq(
                    [(self.b[k, t], 1) for t in range(T)], 1))
        if max_gates is not None:
            self.solver.add(z3.PbLe(
                [(self.gate_sel[gi, t], 1)
                 for gi in range(len(library)) for t in range(T)],
                max_gates))

        # semantics: propagate each flow
        for fi, fl in enumerate(flows):
            xb = [[z3.Bool(f"x{fi}_{t}_{q}") for q in range(nq)]
                  for t in range(T + 1)]
            zb = [[z3.Bool(f"z{fi}_{t}_{q}") for q in range(nq)]
                  for t in range(T + 1)]
            for q in range(nq):
                mode = fl.init_mode.get(q, "fixed")
                if mode == "fixed":
                    self.solver.add(xb[0][q] == (q in fl.init_x),
                                    zb[0][q] == (q in fl.init_z))
                elif mode == "Z":       # start in {I, Z}
                    self.solver.add(z3.Not(xb[0][q]))
                elif mode == "X":       # start in {I, X}
                    self.solver.add(z3.Not(zb[0][q]))
                elif mode == "beta":    # basis chosen by shared selector
                    self.solver.add(
                        z3.Implies(self.beta[q], z3.Not(xb[0][q])),
                        z3.Implies(z3.Not(self.beta[q]),
                                   z3.Not(zb[0][q])))
                elif mode != "free":
                    raise ValueError(f"bad init mode {mode}")
            for t in range(T):
                upd_x = {q: [] for q in range(nq)}
                upd_z = {q: [] for q in range(nq)}
                for gi, g in enumerate(self.library):
                    sel = self.gate_sel[gi, t]
                    if g.name == "CX":     # x_t += x_c ; z_c += z_t
                        c, tg = g.qubits
                        upd_x[tg].append(z3.And(sel, xb[t][c]))
                        upd_z[c].append(z3.And(sel, zb[t][tg]))
                    elif g.name == "CZ":   # z_a += x_b ; z_b += x_a
                        a, bq = g.qubits
                        upd_z[a].append(z3.And(sel, xb[t][bq]))
                        upd_z[bq].append(z3.And(sel, xb[t][a]))
                    elif g.name == "H":    # swap x<->z, as XOR of (x^z)
                        (q,) = g.qubits
                        w = z3.Xor(xb[t][q], zb[t][q])
                        upd_x[q].append(z3.And(sel, w))
                        upd_z[q].append(z3.And(sel, w))
                    elif g.name == "S":    # z += x
                        (q,) = g.qubits
                        upd_z[q].append(z3.And(sel, xb[t][q]))
                for q in range(nq):
                    nx = (z3.Xor(xb[t][q], z3.Or(upd_x[q]))
                          if upd_x[q] else xb[t][q])
                    nz = (z3.Xor(zb[t][q], z3.Or(upd_z[q]))
                          if upd_z[q] else zb[t][q])
                    self.solver.add(xb[t + 1][q] == nx, zb[t + 1][q] == nz)
            # endpoint: data exact; ancillas: flip bit fixed, other bit free
            for q in range(code.n):
                self.solver.add(xb[T][q] == (q in fl.fin_x),
                                zb[T][q] == (q in fl.fin_z))
            for anc, basis in meas_basis.items():
                bit = xb[T][anc] if basis == "Z" else zb[T][anc]
                req = fl.flip.get(anc)
                if req is not None:
                    self.solver.add(bit == bool(req))

    # -- solving ----------------------------------------------------------
    @staticmethod
    def _decided(result):
        """z3 returns sat / unsat / unknown. unknown (resource limit,
        interrupt) must never be silently read as UNSAT -- our UNSAT
        results are used as impossibility proofs."""
        if result == z3.unknown:
            raise RuntimeError("z3 returned 'unknown' (not a proof); "
                               "rerun or raise limits")
        return result == z3.sat

    def check(self):
        return self._decided(self.solver.check())

    def check_assignment(self, assignment):
        """SAT under the assumption that each Tanner edge fires at the given
        layer (used to validate the encoding against brute force)."""
        assumptions = [self.edge_sel[e, t] for e, t in assignment.items()]
        return self._decided(self.solver.check(*assumptions))

    def model_layers(self):
        """Current solution as gate layers: [[(name, qubits), ...], ...]."""
        m = self.solver.model()
        layers = [[] for _ in range(self.T)]
        for gi, g in enumerate(self.library):
            for t in range(self.T):
                if z3.is_true(m[self.gate_sel[gi, t]]):
                    layers[t].append((g.name, g.qubits))
        return layers

    def model_init(self):
        """Chosen reset bases for "beta" qubits: {q: "Z" or "X"}."""
        m = self.solver.model()
        return {q: ("Z" if z3.is_true(m[lit]) else "X")
                for q, lit in self.beta.items()}

    def enumerate(self, n, with_init=False):
        """Solve, BLOCK the found solution, solve again -- until n solutions
        or UNSAT (space exhausted). Solutions arrive in solver order
        (correlated neighborhoods), not as uniform random samples.
        with_init: return (layers, init_basis_map) pairs and block over
        gate AND basis selectors."""
        out = []
        lits = list(self.b.values()) + list(self.beta.values())
        while len(out) < n and self.solver.check() == z3.sat:
            m = self.solver.model()
            out.append((self.model_layers(), self.model_init())
                       if with_init else self.model_layers())
            self.solver.add(z3.Or([
                lit != z3.is_true(m[lit]) for lit in lits]))
        return out


def make_se_synth(code, T, library="sched", max_gates=None,
                  tie_classes=False):
    """tie_classes: tie gates across the code's symmetry orbits
    (code.edge_class, e.g. BB translation classes). Restricts the search to
    symmetric schedules -- the qLDPC analog of the surface code's uniform
    schedules -- collapsing the selector space by the orbit size."""
    # data-preservation flows are provably automatic for the scheduling
    # library (see se_spec docstring); dropping them shrinks big instances.
    flows, basis = se_spec(code, data_flows=(library != "sched"))
    lib = (scheduling_library(code) if library == "sched"
           else clifford_library(code))
    tie = None
    if tie_classes:
        if not hasattr(code, "edge_class"):
            raise ValueError(f"{code.name} has no edge_class metadata")
        tie = (lambda g: code.edge_class[g.edge] if g.edge else None)
    return Synthesizer(code, T, lib, flows, basis,
                       exactly_once_edges=(library == "sched"),
                       max_gates=max_gates, tie=tie)


def find_min_depth(code, library="sched", T0=None, T_max=16,
                   tie_classes=False):
    """Outer budget loop: try depth T, and read UNSAT as a *proof* that no
    circuit of that depth exists; first SAT is depth-optimal. Returns
    (T, synthesizer). With tie_classes, UNSAT proofs apply only to
    orbit-symmetric schedules (tying is a restriction)."""
    if T0 is None:
        from collections import Counter
        T0 = max(Counter(q for _, q in code.edges).most_common(1)[0][1],
                 max(len(s) for _, s, _ in code.checks))
    scope = "symmetric " if tie_classes else ""
    for T in range(T0, T_max + 1):
        synth = make_se_synth(code, T, library, tie_classes=tie_classes)
        if synth.check():
            return T, synth
        print(f"  T={T}: UNSAT (proof: no {scope}depth-{T} circuit exists)")
    raise RuntimeError(f"no circuit up to depth {T_max}")


# ---------------------------------------------------------------------------
# Web-derived hook clauses ("the wind"), surface-code layout only
# ---------------------------------------------------------------------------


def add_surface_hook_clauses(synth, code, d):
    """From the web analysis (webs.py and docs/METHODS.md): the dangerous
    single fault of a weight-4 check is the mid-schedule ancilla fault, whose
    residual is the pair fired in layers {T-1, T}. Forbid X-checks from
    ending on a same-column pair and Z-checks on a same-row pair. Validated
    exact at d=3, T=4: selects all 16 and only the 16 true full-distance
    schedules out of 2704. Scheduling library only (uses edge selectors)."""
    T = synth.T
    for i, (kind, sup, _) in enumerate(code.checks):
        if len(sup) != 4:
            continue
        aligned = [(a, b) for a, b in itertools.combinations(sup, 2)
                   if (a % d == b % d if kind == "X" else a // d == b // d)]
        for qa, qb in aligned:
            for ta, tb in ((T - 2, T - 1), (T - 1, T - 2)):
                synth.solver.add(z3.Not(z3.And(
                    synth.edge_sel[(i, qa), ta],
                    synth.edge_sel[(i, qb), tb])))


# ---------------------------------------------------------------------------
# Beyond syndrome extraction: state preparation
# ---------------------------------------------------------------------------


class _GadgetBox:
    """Minimal stand-in for a code object when the gadget is not SE:
    n data qubits plus n_anc measured ancillas (no Tanner structure)."""

    def __init__(self, n, n_anc, name):
        self.n, self.name = n, name
        self.checks = [("Z", (), n + i) for i in range(n_anc)]


def prep_library(n_data, n_anc, with_h=True):
    """Move set for state preparation: CX between ANY ordered pair of
    qubits (data-data couplings included -- encoding circuits need them),
    plus H on ancillas (so X-type verification is expressible with
    Z-basis-measured ancillas)."""
    qubits = list(range(n_data + n_anc))
    gates = [Gate("CX", (a, b)) for a in qubits for b in qubits if a != b]
    if with_h:
        for a in range(n_data, n_data + n_anc):
            gates.append(Gate("H", (a,)))
    return gates


def prep_spec(code, n_anc):
    """Prepare the all-|0>_L logical state of a CSS code.

    One flow per generator of the target state's stabilizer group (all
    checks + all logical-Z representatives): the generator, as the flow's
    ENDPOINT on data, must be reachable from a product of input reset
    stabilizers (start free within the reset subgroup -- init_mode "beta"
    lets the solver also pick each data qubit's reset basis) times
    measured operators (measurement flip bits left free: the Pauli frame
    may be outcome-conditioned). Verification ancillas start in |0>."""
    n = code.n
    ancs = list(range(n, n + n_anc))
    meas_basis = {a: "Z" for a in ancs}
    targets = []
    for kind, sup, _ in code.checks:
        targets.append((kind, sup))
    for sup in code.logicals_z:
        targets.append(("Z", sup))
    flows = []
    for kind, sup in targets:
        s = frozenset(sup)
        init_mode = {q: "beta" for q in range(n)}
        init_mode.update({a: "Z" for a in ancs})
        flows.append(PauliFlow(
            f"prep target {kind}{sorted(sup)}",
            init_x=frozenset(), init_z=frozenset(),
            fin_x=s if kind == "X" else frozenset(),
            fin_z=s if kind == "Z" else frozenset(),
            flip={a: None for a in ancs},        # frame may use outcomes
            init_mode=init_mode))
    return flows, meas_basis


def make_prep_synth(code, T, n_anc=0, max_gates=None):
    box = _GadgetBox(code.n, n_anc, f"prep({code.name})")
    flows, basis = prep_spec(code, n_anc)
    lib = prep_library(code.n, n_anc)
    return Synthesizer(box, T, lib, flows, basis, max_gates=max_gates,
                       free_init_basis=range(code.n))
