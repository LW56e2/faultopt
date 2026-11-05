"""Common interface for memory-circuit scoring backends.

get_grader("stim") uses heuristic Stim witnesses, which provide upper bounds.
get_grader("webs", cutoff=N) uses the independent signature engine: results
below N are exact, while a result equal to N certifies only a lower bound.
The candidate representation is a list of layers of (gate, qubits) pairs.
"""

from scoring import circuit_from_gates


class StimGrader:
    """Build both memory bases and search for logical-fault witnesses."""

    name = "stim"

    def _distance(self, code, layers, basis, rounds):
        from scoring import circuit_distance
        return circuit_distance(
            circuit_from_gates(code, layers, basis, rounds))

    def measures_code(self, code, layers, rounds=2):
        try:
            circuit_from_gates(code, layers, "Z", rounds,
                               p=0).detector_error_model()
            return True
        except ValueError:
            return False

    def score(self, code, layers, rounds):
        dz, _ = self._distance(code, layers, "Z", rounds)
        dx, _ = self._distance(code, layers, "X", rounds)
        return dz, dx

    def explain(self, code, layers, rounds):
        """Witness of the weaker basis: list of printable fault events."""
        dz, ez = self._distance(code, layers, "Z", rounds)
        dx, ex = self._distance(code, layers, "X", rounds)
        basis, errs = ("Z", ez) if dz <= dx else ("X", ex)
        lines = []
        for e in errs:
            loc = e.circuit_error_locations[0]

            def fmt(t):
                g = t.gate_target
                p = "X" if g.is_x_target else "Y" if g.is_y_target else "Z"
                at = (f"@{t.coords[0]:g},{t.coords[1]:g}"
                      if len(t.coords) >= 2 else "")
                return f"{p}(q{g.qubit_value}{at})"

            targets = " ".join(fmt(t) for t in loc.flipped_pauli_product)
            ndata = sum(1 for t in loc.flipped_pauli_product
                        if t.gate_target.qubit_value < code.n)
            lines.append(f"tick {loc.tick_offset:3d}  "
                         f"{loc.instruction_targets.gate}  -> {targets}"
                         f"   [{ndata} data qubit(s) hit]")
        return basis, lines


_BACKENDS = {"stim": StimGrader}


def get_grader(name="stim", **kwargs):
    """kwargs are backend-specific (e.g. cutoff=N for "webs": distances
    below it are exact, a returned value equal to it means ">= cutoff")."""
    if name not in _BACKENDS:
        try:
            import spectrum  # noqa: F401  registers the "webs" backend
        except ImportError:
            pass
    try:
        return _BACKENDS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown grader '{name}'; "
                         f"available: {sorted(_BACKENDS)}") from None


def register(name, cls):
    """Entry point for future backends (spectrum.py registers 'webs')."""
    _BACKENDS[name] = cls
