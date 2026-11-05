# Implementation and guarantees

## Circuit generation

`synth.py` separates allowed gates from Pauli-flow specifications. The
syndrome-scheduling library uses each Tanner edge once per round with a fixed
CNOT direction, forbids simultaneous gates on the same qubit, and constrains
check readout behavior. State-preparation and richer Clifford specifications
are also available. CEGIS currently uses the exactly-once scheduling library.

The general encoding tracks binary Pauli supports. Signed Clifford action
needs additional care outside the fixed CSS/CNOT skeleton. Solver model order
is not a random-sampling distribution. A solver timeout or `unknown` result
is inconclusive; an enumeration cap is not an impossibility proof. The legacy
`Synth.enumerate` helper does not separately report its termination reason.
Use the checked CEGIS path for its explicit search status.

## Fault signatures

An elementary mechanism is propagated to detector and logical flips. Fault-set
signatures combine by XOR. The minimum-distance search can deduplicate identical
full signatures because using an identical pair cancels its contribution.
Probability calculations and mechanism counts must retain multiplicities.

The independent kernels support exhaustive exclusion below cutoffs through
six. A return equal to the cutoff means only that the distance is at least the
cutoff. The NumPy kernel packs detector bits into machine words; the GPU
weight-five join uses hashes to find candidates and then checks complete keys
and logical effects. GPU direct callers must keep the cutoff at most six.

Stim's `search_for_undetectable_logical_errors` is a heuristic witness search.
The `scoring.circuit_distance` name and legacy `d_z`/`d_x` JSON fields are
retained for compatibility; by themselves those Stim values are witness
weights. An exact claim requires a completed matching lower bound or an
exhaustive below-cutoff result from the independent engine.

## Experiment boundaries and noise

Memory experiments initialize data in the chosen basis, repeat the supplied
schedule, measure-reset ancillas after each round, and finish with data
readout. The fault locations include initialization flips, per-round data
Paulis, all nonidentity Pauli components after noisy gates, ancilla measurement
flips, and final readout flips. Repeated resets are ideal and there is no
additional idle-fault location after each CNOT layer.

These are finite-memory distance statements. They do not alone establish
fault equivalence to an ideal open gadget in arbitrary surrounding protocols.
Open fragments retain local parity responses and boundary residual classes.
`compose_cut` propagates a residual through the next fragment and rebuilds the
complete signatures. General contraction of compressed tables for arbitrary
window optimization remains future work.

## Learned exclusions

CEGIS propagates each witness mechanism individually and collects all qubits
on which it is ever nontrivial. Its clause excludes the joint assignment of
layers to edges incident on that support. It requires at least one of those
gates to move. In the fixed Tanner-edge library and experiment skeleton, fixing
those edge layers transports each mechanism and its detector/logical effects.
This argument does not automatically extend to gate insertions, removals, or
schedule-dependent new fault locations.
