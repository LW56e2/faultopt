"""Post-selected injection analysis and the optional Lao-Criger sweep.

For the injection protocols tested here, the objective is a census: how many fault mechanisms are accepted by post-selection yet
corrupt the injected logical state (weight 1 and weight-2 pairs), and
the probability-weighted coefficients (p_L ~ c1*p + c2*p^2 under the
location model of inject.py, post-selected on zero detection events).

The non-Clifford input is scored through Clifford stand-in frames; all
declared parities (round-1 detectors, growth detectors, teleportation
corrections) are DERIVED by the pipeline and validated against stim at
p=0; every weight-1 count is cross-checked against stim's detector error
model. Everything printed is computed on the spot; pinned expectations
live in tests/test_inject.py.
"""

import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

from codes import (bb_code, low_weight_logical_reps, surface_css,
                   uniform_assignment)
from scoring import layers_from_assignment
from inject import (GOOD_SCHED, hook_frames, injection_report,
                    standard_frames, surface_pattern, zcheck_hook_config,
                    zcheck_hook_frames)

D = 3
BAD_SCHED = (("NW", "NE", "SW", "SE"), ("NW", "SW", "NE", "SE"))
WIN1 = (("NE", "SE", "NW", "SW"), ("NE", "NW", "SE", "SW"))


def line(tag, report, extra=""):
    fr = report["frames"]
    if all("stim_agrees" in f for f in fr):
        stim_ok = all(f["stim_agrees"] for f in fr)
    else:
        stim_ok = "unchecked"
    per = " ".join(
        f"{f['frame']}:w1={f['w1_count']}/c1={f['w1_weighted']}"
        f"/w2={f['w2_pairs']}" for f in fr)
    print(f"{tag} {per} union_w1={report['union']['w1_count']} "
          f"union_c1={report['union']['w1_weighted']} "
          f"kept={report['detectors_kept']} "
          f"dropped={len(report['detectors_dropped'])} "
          f"stim_agrees={stim_ok}{extra}")


def main():
    code, raw = surface_css(D)
    layers = {
        "goodSE": layers_from_assignment(
            code, uniform_assignment(raw, *GOOD_SCHED), 4),
        "win1SE": layers_from_assignment(
            code, uniform_assignment(raw, *WIN1), 4),
        "badSE": layers_from_assignment(
            code, uniform_assignment(raw, *BAD_SCHED), 4),
    }

    print("== standard injection (site x pattern), goodSE growth,"
          " 2 SE rounds ==")
    for (r, c) in ((0, 0), (1, 1), (2, 2)):
        for style in ("diag", "col"):
            im = surface_pattern(D, r, c, style)
            init = "".join(im.get(q, ".") for q in range(code.n))
            try:
                frames = standard_frames(code, layers["goodSE"],
                                         r * D + c, im, 2)
                rep, _, _ = injection_report(code, frames)
                line(f"standard site=({r},{c}) pattern={style} "
                     f"init={init}", rep)
            except ValueError as e:
                print(f"standard site=({r},{c}) pattern={style} "
                      f"init={init} rejected={e}")

    print()
    print("== hook injection (anchor 0) x hook column x growth schedule ==")
    for sched in ("goodSE", "win1SE", "badSE"):
        for head_first in (True, False):
            frames, pair = hook_frames(code, D, layers[sched], 2,
                                       head_first=head_first)
            rep, _, _ = injection_report(code, frames)
            line(f"hook growth={sched} pair={pair}", rep)

    print()
    print("== hook position / orientation (anchor 0): only the subpatch"
          " logical injects ==")
    for tail in ((0, 3), (1, 4), (0, 1), (3, 4), (0,), (1, 3, 4)):
        try:
            frames, pair = hook_frames(code, D, layers["goodSE"], 2,
                                       tail=tail)
            rep, _, _ = injection_report(code, frames)
            line(f"hook tail={tail}", rep)
        except ValueError as e:
            print(f"hook tail={tail} rejected={e}")

    print()
    print("== d=5: compare the weight-1 hook census with d=3 ==")
    code5, raw5 = surface_css(5)
    L5 = layers_from_assignment(code5, uniform_assignment(raw5, *WIN1), 4)
    frames = standard_frames(code5, L5, 0, surface_pattern(5, 0, 0, "diag"),
                             2)
    rep, _, _ = injection_report(code5, frames)
    line("d=5 standard site=(0,0) diag", rep)
    frames, pair = hook_frames(code5, 5, L5, 2)
    rep, _, _ = injection_report(code5, frames)
    line(f"d=5 hook pair={pair}", rep)

    print()
    print("== best hook configuration: the corrupting mechanisms ==")
    frames, pair = hook_frames(code, D, layers["goodSE"], 2)
    rep, _, _ = injection_report(code, frames)
    print(f"hook_pair={pair} init="
          + "".join(frames[0]["init_map"].get(q, "Z")
                    for q in range(code.n)))
    for f in rep["frames"]:
        print(f"{f['frame']} observable={f['observable']} "
              f"folded_records={f['folded_records']}")
    for lab in rep["union"]["w1_labels"]:
        print(f"union_w1_mechanism {lab}")

    print()
    print("== Lao-Criger CF'22 criterion (|Y> state, their model: no"
          " idles) ==")
    print("== their Eq2 CR: 9/15 p2 + 2 pIN; Eq3 MR: 9/15 p2 + 1 pIN"
          " (+ 2/3 p1 rotation gate, not a location here) ==")
    from inject import lc_census
    CR = {0: "Z", 1: "Z", 3: "Z", 4: "X", 5: "X", 6: "X", 7: "X", 8: "X"}
    MR = {0: "Z", 1: "X", 2: "X", 3: "Z", 5: "Z", 6: "X", 7: "X", 8: "Z"}
    wins = {"goodSE": layers["goodSE"], "win1SE": layers["win1SE"]}
    for tag, site, im in (("CR(site=q2,mirrored)", 2, CR),
                          ("MR(center)", 4, MR)):
        for sname, L in wins.items():
            c = lc_census(code, L, site, im)
            print(f"lc {tag} sched={sname} p2={c['p2_fifteenths']}/15 "
                  f"pIN={c['p_in']} pM={c['p_m']} p1={c['p1_thirds']}/3 "
                  f"mismatches={c['frame_detect_mismatches']}")

    print()
    print("== BB [[72,12,6]]: qLDPC hook injection (growth = the CEGIS"
          " depth-7 schedule) ==")
    bb = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)],
                 name="BB [[72,12,6]]")
    with open(ARTIFACTS / "bb72_cegis_winners.json") as f:
        growth = [[(n, tuple(q)) for n, q in layer]
                  for layer in json.load(f)["circuits"][-1]]
    lz = low_weight_logical_reps(bb, "Z")[0]
    ci, tail, lx = zcheck_hook_config(bb, lz)
    print(f"config check={ci} tail={tail} lz={lz} lx={lx}")
    for mode in ("inj", "xsafe"):
        frames = zcheck_hook_frames(bb, ci, tail, lz, lx, growth, 2,
                                    r0_checks=mode)
        rep, _, _ = injection_report(bb, frames, stim_check=False)
        line(f"bb72 hook r0={mode}", rep)


LC_CR = {0: "Z", 1: "Z", 3: "Z", 4: "X", 5: "X", 6: "X", 7: "X", 8: "X"}
LC_MR = {0: "Z", 1: "X", 2: "X", 3: "Z", 5: "Z", 6: "X", 7: "X", 8: "Z"}


def _lc_worker(args):
    from inject import lc_census
    idx, layers = args
    code, _ = surface_css(3)
    out = {"i": idx}
    for tag, site, im in (("CR", 2, LC_CR), ("MR", 4, LC_MR)):
        try:
            c = lc_census(code, [list(l) for l in layers], site, dict(im))
            out[tag] = (c["p2_fifteenths"], c["p_in"], c["p_m"])
        except ValueError:
            out[tag] = None
    return out


def lc_sweep(jobs=8):
    """Exhaustive: every valid depth-4 per-check schedule (the 2704 of
    the recorded surface-code enumeration), scored under the Lao-Criger criterion for both of their
    protocols. Open-ended runtime (~20 min, 8 workers)."""
    from collections import Counter
    from multiprocessing import Pool

    from synth import make_se_synth
    code, raw = surface_css(3)
    synth = make_se_synth(code, 4, "sched")
    cands = synth.enumerate(5000)
    print(f"schedules={len(cands)}")
    with Pool(jobs) as pool:
        rows = pool.map(_lc_worker, list(enumerate(cands)))
    for tag, lc_claim in (("CR", 9), ("MR", 9)):
        vals = [r[tag] for r in rows if r[tag] is not None]
        hist = dict(sorted(Counter(v[0] for v in vals).items()))
        best = min(v[0] for v in vals)
        print(f"lc_sweep {tag} injectable={len(vals)} "
              f"p2_histogram={hist} "
              f"pIN={dict(Counter(v[1] for v in vals))} "
              f"pM={dict(Counter(v[2] for v in vals))} "
              f"minimum={best}/15 "
              f"n_at_minimum={sum(1 for v in vals if v[0] == best)} "
              f"n_at_lc_value={sum(1 for v in vals if v[0] == lc_claim)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lc-sweep", action="store_true",
                        help="run the expensive per-check schedule census")
    parser.add_argument("--jobs", type=int, default=8,
                        help="workers for --lc-sweep (default: 8)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.lc_sweep:
        lc_sweep(args.jobs)
    else:
        main()
