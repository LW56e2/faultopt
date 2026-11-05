# Reproduction and evidence

Run commands from an installed source checkout. Keep generated output in
`runs/`. New SAT/CEGIS circuits may differ from the saved ones because solver
model order is not fixed. Original JSON payloads are retained in `artifacts/`;
`artifacts/manifest.json` records their checksums and origin filenames.

## Steane memory circuit

```sh
python scripts/verify_memory.py --code steane \
  --circuit artifacts/steane_t6_winner.json --rounds 3 --cutoff 4 --stim
```

Repeat with `--rounds 4` and `--rounds 5`. The expected exact value is three in
both bases for each of these finite experiments. The independent cutoff-four
search excludes smaller mechanisms and finds weight three; Stim supplies a
matching witness. The schedule has six CNOT layers and 24 CNOTs with six
concurrent bare check ancillas. Those resources are minimal within the direct
Tanner-edge model, not across every error-correction architecture.

## Surface-code enumeration

```sh
fto sweep --code surface:3 --jobs 1 --out runs/surface3.json
fto cegis --code surface:3 -T 4 --target 3 --exhaust \
  --save runs/surface3-winners.json
```

Of 576 uniform corner-permutation pairs, 96 satisfy parallelism, 16 are
semantically valid, and four attain full distance in the stored distance-three
and distance-five sweeps. The stored sweeps contain the 96 parallel schedules;
nonmeasuring schedules have null distance fields. Their `d_z`/`d_x` values
originate from the Stim witness search, so the files alone are not exhaustive
distance certificates. The independent scorer and regression tests supply
additional checks.

The complete per-check depth-four space contains 2704 valid schedules, with 16
distance-three winners. The recorded CEGIS enumeration scored 136 candidates
and learned 186 cone clauses. Iteration counts are solver-order dependent.

## BB72

`artifacts/bb72_t7_tied_d4.json` is the initial depth-seven baseline;
`artifacts/bb72_cegis_winners.json` is the saved target-five CEGIS winner.
Both results use two-round memories. The project campaign records report
completed CPU and GPU exclusions below six for the winner, together with
weight-six Stim witnesses: these establish exact distance six in both bases.
The original target five did not limit the distance ultimately verified.

The historical kernel comparison recorded BB72 CPU exclusions returning six
in both bases (376/455 seconds with 192 workers, and 7129 seconds on eight
laptop cores) and GPU agreement at about 79 seconds per basis. These are
archived observations on different machines, not controlled benchmarks.
The original full execution logs and independent proof traces are not bundled
here; the saved circuit and reproduction commands are. The artifact's original
note calls the Stim result "exact"; interpret the Stim result as an upper-bound
witness and use the completed signature exclusions for the lower bound.

To rederive the threshold on the stored circuit, use a machine with sufficient
memory and, optionally, a CUDA GPU:

```sh
python scripts/verify_memory.py --code bb72 \
  --circuit artifacts/bb72_cegis_winners.json --rounds 2 \
  --cutoff 6 --kernel numpy --procs 1 --stim
# On a suitable CUDA/CuPy host, replace --kernel numpy with --kernel gpu.
```

Cutoff-six joins can take substantial time and memory. Interrupting the search
does not produce a lower-bound certificate. To rerun the restricted depth
feasibility checks:

```sh
fto synth --code bb72 -T 6 --tie
fto synth --code bb72 -T 6 --tie --tie-split 2
fto synth --code bb72 -T 6 --tie --tie-split 3
```

The tested families have 12, 48, and 108 classes. Their depth-six UNSAT reports
are restricted to those families; unrestricted asymmetric depth six is open
in this project.

## Gross code

The target-five and target-six CEGIS reports contain the same seven-layer,
864-CNOT schedule. The target-six GPU report records a lower bound of six in
both bases at two rounds. It is not an exact distance for the nominal
`[[144,12,12]]` code.

```sh
python scripts/verify_memory.py --code gross \
  --circuit artifacts/gross_cegis_t6_gpu.json --rounds 2 \
  --cutoff 6 --kernel gpu
fto synth --code gross -T 6 --tie
fto synth --code gross -T 6 --tie --tie-split 2
fto synth --code gross -T 6 --tie --tie-split 3
```

The three tiny `gross_T6_*.json` files are archived solver reports, not
independently checkable proof traces. They have identical contents but refer
to different families through their filenames; all three are retained.
The large `gross_T7_tied_first.json` also preserves the initially generated
candidates. Large GPU exclusions and all SAT proofs are not rerun by default.

## Preparation and injection

```sh
fto prep --code steane --ft
fto inject --code surface:3 --scheme hook --rounds 2
python examples/injection_analysis.py --help
# Optional long census:
python examples/injection_analysis.py --lc-sweep --jobs 8
```

The depth-three, eight-CNOT bare Steane encoder is not a verified
fault-tolerant preparation protocol. Injection output counts accepted
corruption mechanisms under the implemented model; it is not a decoder
logical error rate or a noise-model-independent comparison.
