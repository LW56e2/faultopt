# faultopt

`faultopt` synthesizes and checks quantum syndrome-extraction schedules.
It combines Pauli-flow specifications, Z3 circuit generation, fault-signature
search, and counterexample-guided inductive synthesis (CEGIS).

The associated paper is *Fault Tolerance as an Objective Function: Certified
Synthesis of Syndrome-Extraction Circuits* (preprint soon). The principal experiments use
rotated surface codes, Steane, BB72, and the gross bivariate bicycle code.

## Install

Use Python 3.12 or newer. From this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]" -c requirements-repro.txt
python -m pytest
```

The constraints file records the direct runtime and test versions used in
local checks. It is not a complete transitive dependency lock. The CPU suite
was checked locally with Python 3.13.5; GitHub Actions is configured for
Python 3.12 and 3.13. GPU calculations require a compatible NVIDIA CUDA/CuPy
installation and are not part of the default tests. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Quick start

```sh
# Inspect a built-in code.
fto describe --code steane

# Independently verify the stored Steane memory circuit in both bases.
python scripts/verify_memory.py --code steane \
  --circuit artifacts/steane_t6_winner.json --rounds 3 --cutoff 4 --stim

# Enumerate all qualifying depth-four surface-code schedules.
fto cegis --code surface:3 -T 4 --target 3 --exhaust

# Synthesize a Steane schedule; the result can depend on solver model order.
mkdir -p runs
fto cegis --code steane -T 6 --target 3 --save runs/steane.json
```

`fto --help` lists the commands: `describe`, `synth`, `cegis`, `score`,
`sweep`, `window`, `cut`, `prep`, `inject`, and `viz`. The original
`python fto.py ...` invocation also works from the checkout.

## Reading the results

The independent signature engines search exhaustively **below** a cutoff.
A returned value below the cutoff is an exact distance; a value equal to the
cutoff certifies a lower bound. Supported cutoffs are at most six. Every
required join must finish before a lower bound is claimed.

The default `fto score` backend uses Stim's heuristic witness search. A
weight-*w* witness is an upper bound, not an exhaustive lower-bound proof.
Use `--scorer webs --cutoff 4` for independent small-instance scoring, or
`scripts/verify_memory.py` to select the Python, NumPy, or GPU kernel and
report the bound explicitly.

All circuit distances refer to the specified memory basis, number of rounds,
fault locations, initialization, and final readout. Depth counts CNOT layers.
The noise model omits per-layer idle faults and faults on repeated ancilla
resets. The Steane resource optimum is within exactly-once Tanner-edge
scheduling; bicycle depth impossibility is restricted to the tested symmetry
families.

[Reproduction and evidence](docs/REPRODUCIBILITY.md) distinguishes archived
large runs from checks that can be reproduced quickly. [Methods](docs/METHODS.md)
describes the implementation and its limits.

## Repository layout

| Path | Contents |
| --- | --- |
| `fto.py` | Command-line interface and JSON input/output |
| `codes.py`, `f2.py` | CSS code definitions and binary linear algebra |
| `synth.py`, `cegis.py` | Pauli-flow constraints and witness-guided search |
| `scoring.py`, `grader.py` | Stim circuit construction and backend interface |
| `webs.py`, `spectrum.py` | Pauli webs, open fragments, and signature search |
| `mim.py`, `mim_gpu.py` | NumPy and optional CuPy threshold kernels |
| `inject.py`, `viz.py` | Injection analysis and circuit visualization |
| `artifacts/` | Saved circuits, threshold reports, and surface sweeps |
| `scripts/` | Stored-memory verification and paper-figure generators |
| `examples/` | Additional injection analysis and an optional expensive sweep |
| `tests/` | CPU regressions and explicitly marked slow checks |

Core modules retain their original flat import layout. The CLI can be
installed independently, while the saved artifacts, examples, and figure
scripts are supplied in the source checkout.

## Tests and figures

```sh
python -m pytest                  # fast CPU suite
python -m pytest -m slow          # expensive SAT and large-code checks
python scripts/paper_fig_satgrid.py
python scripts/paper_fig_steane.py
python examples/injection_analysis.py --help
```

The slow suite can require substantial time and memory. Run it deliberately;
it is not part of default CI. Figure generators write SVG and Quantikz
sources into `figures/`. The optional injection example retains analyses
that are not covered by the main CLI.

## License

Released under the [MIT License](LICENSE).
