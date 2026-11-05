# Saved research artifacts

The JSON payloads are preserved byte for byte from the working research
checkout. See `manifest.json` for checksums and original filenames.

| File | Purpose |
| --- | --- |
| `steane_t6_winner.json` | Six-layer, 24-CNOT Steane schedule |
| `bb72_t7_tied_d4.json` | Initial tied depth-seven BB72 baseline |
| `bb72_cegis_winners.json` | Saved BB72 CEGIS winner |
| `gross_T6_tied.json` | Fully tied depth-six UNSAT report |
| `gross_T6_split2.json` | Split-2 depth-six UNSAT report |
| `gross_T6_split3.json` | Split-3 depth-six UNSAT report |
| `gross_T7_tied_first.json` | Initial depth-seven gross-code candidates |
| `gross_cegis_t5_run1.json` | Gross-code target-five CEGIS report |
| `gross_cegis_t6_gpu.json` | Gross-code target-six GPU CEGIS report |
| `lc_cr_optimum.json`, `lc_mr_optimum.json` | Saved injection-census schedules |
| `surface_d3_uniform_sweep.json` | Stored surface distance-three sweep |
| `surface_d5_uniform_sweep.json` | Stored surface distance-five sweep |

Matching circuit payloads can represent different threshold experiments.
The three matching UNSAT payloads refer to distinct symmetry families. These
files therefore are not interchangeable duplicates.

Historical notes and field names are retained, including old uses of "exact"
for Stim. Consult [reproduction and evidence](../docs/REPRODUCIBILITY.md) for
the correct interpretation. A JSON report of a computation is not itself an
independently checkable proof trace. Record the rounds, memory basis, noise
model, cutoff, kernel, and completion status when generating new verification
results.
