"""Verify a saved finite-memory circuit with explicit bound semantics.

Run from an installed source checkout. Large cutoff-six jobs can require
substantial memory and time; an interrupted job produces no certificate.
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from types import SimpleNamespace

from fto import load_circuits, load_code
from scoring import circuit_distance, circuit_from_gates
from spectrum import closed_signatures, declared_parities, fragment_from_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="built-in code or CSS JSON file")
    parser.add_argument("--circuit", required=True, help="saved circuit JSON file")
    parser.add_argument("--index", type=int, default=0, help="candidate index (default: 0)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cutoff", type=int, choices=range(2, 7), default=4)
    parser.add_argument("--kernel", choices=["python", "numpy", "gpu"], default="numpy")
    parser.add_argument("--procs", type=int, default=1, help="NumPy weight-five workers")
    parser.add_argument("--stim", action="store_true", help="also search for a Stim upper-bound witness")
    args = parser.parse_args()
    if args.rounds < 1 or args.procs < 1 or args.index < 0:
        parser.error("rounds/procs must be positive and index must be nonnegative")

    code = load_code(args.code)
    candidates = load_circuits(SimpleNamespace(circuit=args.circuit, uniform=None), code)
    if args.index >= len(candidates):
        parser.error("candidate index is outside the saved circuit list")
    layers = candidates[args.index]
    if args.kernel == "python":
        from spectrum import min_weight
        search = lambda sigs, n_det: min_weight(sigs, n_det, args.cutoff)
    elif args.kernel == "numpy":
        from mim import min_weight_packed
        search = lambda sigs, n_det: min_weight_packed(sigs, n_det, args.cutoff, procs=args.procs)
    else:
        try:
            from mim_gpu import min_weight_gpu
        except ImportError:
            parser.error("GPU verification requires a compatible CuPy/CUDA installation")
        search = lambda sigs, n_det: min_weight_gpu(sigs, n_det, args.cutoff)

    result = {
        "code": code.name, "candidate_index": args.index,
        "rounds": args.rounds, "cutoff": args.cutoff, "kernel": args.kernel,
        "versions": {"python": platform.python_version(),
                     **{name: version(name) for name in ("numpy", "stim", "z3-solver")}},
        "bases": {},
    }
    for basis in "ZX":
        start = time.monotonic()
        circuit = circuit_from_gates(code, layers, basis, args.rounds)
        circuit.detector_error_model()  # Reject nondeterministic detector declarations.
        fragment = fragment_from_layers(code, layers, args.rounds, basis, basis)
        detectors, observables = declared_parities(code, args.rounds, basis)
        signatures, _ = closed_signatures(fragment, detectors, observables)
        value = int(search(signatures, len(detectors)))
        row = {
            "value": value,
            "kind": "exact_distance" if value < args.cutoff else "lower_bound",
            "search_completed": True,
            "excluded_weights_below": value,
            "elementary_mechanisms": len(signatures),
            "distinct_signatures": len(set(signatures)),
        }
        if args.stim:
            try:
                weight, _ = circuit_distance(circuit)
            except ValueError as error:
                row["stim_search_error"] = str(error)
            else:
                if weight < value:
                    raise RuntimeError("Stim witness contradicts the independent lower bound")
                row["stim_upper_bound"] = weight
                if weight == value:
                    row["kind"] = "exact_distance"
        row["seconds"] = round(time.monotonic() - start, 3)
        result["bases"][basis] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
