#!/usr/bin/env python3
"""fto -- one entry point for the whole toolkit, with uniform JSON I/O.

Every command takes --code and prints a JSON result (or --out FILE).

CODE SPECS (--code):
    surface:3            rotated surface code, distance 3 (any odd d)
    steane               [[7,1,3]]
    bb72                 bivariate bicycle [[72,12,6]]
    gross                bivariate bicycle l=12,m=6 (the [[144,12,12]] family)
    path/to/code.json    {"name": ..., "n": int, "hx": [[q,..],..], "hz": ...}

CIRCUIT FILES (--circuit, produced by `synth --save`):
    {"code": name, "T": int, "circuits": [ [ [ ["CX",[c,t]], ...], ...], ...]}
    (a list of candidates; each candidate is a list of layers of gates)

COMMANDS
    describe                         code parameters, checks, logicals
    prep     [-T auto|N] [--anc M] [--enum K] [--ft] [--save FILE]
             synthesize state-preparation circuits (|0>_L) and optionally
             fault-tolerance-check them with the open scorer
    synth    [-T auto|N] [--library sched|clifford] [--tie] [--hookfree]
             [--max-gates N] [--enum N] [--save FILE]
    cegis    --target D [-T auto|N] [--rounds N] [--tie] [--exhaust]
             [--max-iters N] [--full] [--save FILE]
             counterexample-guided search for schedules with d_eff >= D;
             exhausting the space (status "exhausted") is a PROOF that no
             further such schedule exists at this depth
    score    (--circuit FILE | --uniform "NW>SW>NE>SE/NW>NE>SW>SE")
             [--rounds N] [--scorer stim|webs] [--explain]
    sweep    [--jobs N] [--rounds N] [--scorer]     (surface codes only)
    window   (--circuit | --uniform) [--rounds N]   open-window spectrum
    cut      (--circuit | --uniform)                cut test vs uncut stim
    inject   --scheme standard|hook [--site r,c | --anchor Q]
             [--pattern diag|col|row] [--init ZX..] [--rounds N]
             [--uniform ../..] [--circuit FILE] [--no-stim-check]
             [--crumble]      magic-state injection corruption census
             (surface codes; frames, declared parities, and teleport
             corrections are derived and validated mechanically)
    viz      (--circuit | --uniform) [--rounds N] [--basis Z|X]
             --type timeline|timeslice|detslice|matchgraph|crumble
             [--noise] [--out FILE.svg]     render the scored circuit

EXAMPLES
    python fto.py describe --code gross
    python fto.py synth --code steane -T auto --enum 20 --save cand.json
    python fto.py cegis --code surface:3 --target 3 --exhaust
    python fto.py score --code steane --circuit cand.json --rounds 3
    python fto.py score --code surface:3 --uniform \\
        "NW>SW>NE>SE/NW>NE>SW>SE" --scorer webs --explain
    python fto.py cut --code surface:3 --uniform "NW>SW>NE>SE/NW>SW>NE>SE"
"""

import argparse
import json
import sys
import time
from collections import Counter

from codes import (CORNERS, bb_code, css_from_matrices, schedule_valid,
                   split_translation_classes, steane_css, surface_css,
                   uniform_assignment)
from grader import get_grader
from scoring import layers_from_assignment
from synth import add_surface_hook_clauses, find_min_depth, make_se_synth

MONO_A = [(3, 0), (0, 1), (0, 2)]
MONO_B = [(0, 3), (1, 0), (2, 0)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_code(spec):
    if spec.startswith("surface:"):
        d = int(spec.split(":", 1)[1])
        code, raw = surface_css(d)
        code._raw, code._d = raw, d
        return code
    if spec == "steane":
        return steane_css()
    if spec == "bb72":
        return bb_code(6, 6, MONO_A, MONO_B, name="BB [[72,12,6]]")
    if spec == "gross":
        return bb_code(12, 6, MONO_A, MONO_B, name="gross BB(12,6)")
    with open(spec) as f:
        j = json.load(f)
    return css_from_matrices(j.get("name", spec), j["hx"], j["hz"], j["n"])


def load_circuits(args, code):
    """Candidate gate-layer lists, from --circuit or --uniform."""
    if args.circuit:
        with open(args.circuit) as f:
            j = json.load(f)
        raw = j["circuits"] if "circuits" in j else [j["layers"]]
        return [[[(name, tuple(qs)) for name, qs in layer]
                 for layer in cand] for cand in raw]
    if args.uniform:
        if not hasattr(code, "_raw"):
            sys.exit("--uniform needs a surface code (--code surface:d)")
        sz, sx = (tuple(p.split(">")) for p in args.uniform.split("/"))
        for p in (sz, sx):
            if sorted(p) != sorted(CORNERS):
                sys.exit(f"bad permutation {p}; corners are {CORNERS}")
        return [layers_from_assignment(
            code, uniform_assignment(code._raw, sz, sx), 4)]
    sys.exit("need --circuit FILE or --uniform SZ/SX")


def emit(obj, out):
    text = json.dumps(obj, indent=1, default=list)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_describe(args):
    code = load_code(args.code)
    from collections import Counter as C
    deg = C(q for _, q in code.edges)
    emit({
        "name": code.name,
        "n_data": code.n,
        "k": getattr(code, "k", len(code.logicals_z)),
        "check_weight_histogram": {
            "X": dict(C(len(s) for k, s, _ in code.checks if k == "X")),
            "Z": dict(C(len(s) for k, s, _ in code.checks if k == "Z"))},
        "data_degree_max": max(deg.values()),
        "logical_weights": {
            "Z": sorted(len(s) for s in code.logicals_z),
            "X": sorted(len(s) for s in code.logicals_x)},
        "has_translation_classes": hasattr(code, "edge_class"),
    }, args.out)


def _apply_tie_split(args, code):
    if getattr(args, "tie_split", None):
        if not args.tie:
            sys.exit("--tie-split requires --tie")
        if not hasattr(code, "edge_class"):
            sys.exit(f"{code.name} has no translation classes")
        split_translation_classes(code, args.tie_split)


def cmd_synth(args):
    code = load_code(args.code)
    _apply_tie_split(args, code)
    t0 = time.time()
    if args.T == "auto":
        T, synth = find_min_depth(code, library=args.library,
                                  tie_classes=args.tie)
    else:
        T = int(args.T)
        synth = make_se_synth(code, T, args.library,
                              max_gates=args.max_gates,
                              tie_classes=args.tie)
        if not synth.check():
            emit({"T": T, "sat": False,
                  "note": "no circuit at this depth (proof)"}, args.out)
            return
    if args.hookfree:
        if not args.code.startswith("surface:"):
            sys.exit("--hookfree is surface-specific")
        add_surface_hook_clauses(synth, code, code._d)
        if not synth.check():
            sys.exit("hook clauses infeasible at this depth")
    circuits = synth.enumerate(args.enum) if args.enum else \
        [synth.model_layers()]
    result = {"code": code.name, "T": T, "library": args.library,
              "tie": args.tie, "n_circuits": len(circuits),
              "seconds": round(time.time() - t0, 1),
              "circuits": circuits}
    emit(result, args.out or args.save)


def cmd_cegis(args):
    from cegis import cegis_search
    code = load_code(args.code)
    _apply_tie_split(args, code)
    if args.T == "auto":
        T, _ = find_min_depth(code, tie_classes=args.tie)
    else:
        T = int(args.T)
    res = cegis_search(code, T, args.target, rounds=args.rounds,
                       tie_classes=args.tie, exhaust=args.exhaust,
                       max_iters=args.max_iters, verbose=args.verbose,
                       kernel=args.kernel, procs=args.procs)
    out = {"code": code.name}
    out.update({k: res[k] for k in
                ("status", "T", "d_target", "rounds", "tie", "iterations",
                 "learned_clauses", "proof", "seconds")})
    if res["proof"]:
        scope = "orbit-symmetric " if args.tie else ""
        out["note"] = (f"proof: no {scope}depth-{T} schedule beyond the "
                       f"{len(res['winners'])} returned reaches d_eff >= "
                       f"{args.target} (noise model of scoring."
                       f"circuit_from_gates, rounds={args.rounds})")
    out["winner_scores"] = [w["scores"] for w in res["winners"]]
    out["circuits"] = [w["layers"] for w in res["winners"]]
    if args.full:
        out["patterns"] = res["patterns"]
        out["log"] = res["log"]
    emit(out, args.out or args.save)


def cmd_prep(args):
    from synth import make_prep_synth
    from scoring import prepares_state
    from spectrum import prep_fragment, prep_ft_report
    code = load_code(args.code)
    t0 = time.time()
    if args.T == "auto":
        for T in range(1, 10):
            synth = make_prep_synth(code, T, args.anc,
                                    max_gates=args.max_gates)
            if synth.check():
                break
            print(f"T={T}: UNSAT (proof)", file=sys.stderr)
    else:
        T = int(args.T)
        synth = make_prep_synth(code, T, args.anc,
                                max_gates=args.max_gates)
        if not synth.check():
            emit({"T": T, "sat": False}, args.out)
            return
    ancs = list(range(code.n, code.n + args.anc))
    rows = []
    for layers, init in synth.enumerate(max(args.enum, 1), with_init=True):
        row = {"T": T, "gates": sum(len(l) for l in layers),
               "init": "".join(init.get(q, "Z") for q in range(code.n)),
               "layers": layers}
        if args.anc == 0:
            row["prepares_state"] = prepares_state(code, layers, init)
        if args.ft:
            rep = prep_ft_report(
                prep_fragment(code.n, layers, init, ancs), code.n)
            row["ft_violations"] = len(rep["violations"])
            row["fault_tolerant_d3"] = rep["fault_tolerant_d3"]
        rows.append(row)
    emit({"code": code.name, "n_anc": args.anc,
          "seconds": round(time.time() - t0, 1),
          "candidates": rows}, args.out or args.save)


def cmd_score(args):
    code = load_code(args.code)
    cands = load_circuits(args, code)
    kw = {}
    if args.cutoff is not None:
        if args.scorer != "webs":
            sys.exit("--cutoff is a webs-scorer knob")
        kw["cutoff"] = args.cutoff
    g = get_grader(args.scorer, **kw)
    rows = []
    for i, layers in enumerate(cands):
        t0 = time.time()
        try:
            if args.census:
                from inject import closed_census
                row = {"i": i}
                for basis in ("Z", "X"):
                    c = closed_census(code, layers, args.rounds, basis)
                    row[f"census_{basis}"] = {
                        k: c[k] for k in
                        ("mechanisms", "detected", "w1_count",
                         "w1_weighted", "w2_pairs", "w2_weighted")}
                row["seconds"] = round(time.time() - t0, 2)
                rows.append(row)
                continue
            dz, dx = g.score(code, layers, args.rounds)
            row = {"i": i, "d_z": dz, "d_x": dx, "d_eff": min(dz, dx),
                   "seconds": round(time.time() - t0, 2)}
            if args.explain:
                basis, witness = g.explain(code, layers, args.rounds)
                row["witness_basis"] = basis
                row["witness"] = witness
        except ValueError as e:
            row = {"i": i, "error": f"does not measure the code ({e})"}
        rows.append(row)
    emit({"code": code.name, "scorer": args.scorer, "rounds": args.rounds,
          "results": rows}, args.out)


def _sweep_worker(job):
    d, rounds, sz, sx, scorer = job
    code, raw = surface_css(d)
    layers = layers_from_assignment(code, uniform_assignment(raw, sz, sx), 4)
    g = get_grader(scorer)
    try:
        dz, dx = g.score(code, layers, rounds)
        return {"sz": sz, "sx": sx, "d_z": dz, "d_x": dx,
                "d_eff": min(dz, dx)}
    except ValueError:
        return {"sz": sz, "sx": sx, "d_eff": None}


def cmd_sweep(args):
    import itertools
    from multiprocessing import Pool
    if not args.code.startswith("surface:"):
        sys.exit("sweep is for surface codes")
    code = load_code(args.code)
    d = code._d
    rounds = args.rounds or d
    perms = list(itertools.permutations(CORNERS))
    jobs = [(d, rounds, sz, sx, args.scorer)
            for sz in perms for sx in perms
            if schedule_valid(code._raw, sz, sx)]
    t0 = time.time()
    with Pool(args.jobs) as pool:
        rows = pool.map(_sweep_worker, jobs)
    hist = Counter(r["d_eff"] for r in rows)
    emit({"code": code.name, "considered": len(perms) ** 2,
          "parallel_valid": len(jobs),
          "histogram": {str(k): v for k, v in sorted(
              hist.items(), key=lambda kv: str(kv[0]))},
          "winners": [r for r in rows if r["d_eff"] == d],
          "seconds": round(time.time() - t0, 1)}, args.out)


def cmd_window(args):
    from spectrum import (fragment_from_layers, restrict_undetectable,
                          spectrum_table, window_signatures)
    code = load_code(args.code)
    layers = load_circuits(args, code)[0]
    fr = fragment_from_layers(code, layers, args.rounds, "Z", None)
    _, dets = fr.detector_basis()
    local = [m for _, m in dets]
    wsigs, piv, order = window_signatures(fr, local,
                                          list(range(code.n)))
    T = spectrum_table([s for s, _ in wsigs])
    mask = (1 << len(local)) - 1
    R = restrict_undetectable(T, mask)
    weight1 = sorted({lab for s, lab in wsigs
                      if s and not (s & mask)})
    emit({"code": code.name, "rounds": args.rounds,
          "open_data_legs": code.n,
          "internal_detectors": len(local),
          "spectrum_entries": len(T),
          "window_undetectable_entries": len(R),
          "weight1_undetectable_mechanisms": len(weight1),
          "examples": weight1[:8]}, args.out)


def cmd_cut(args):
    from spectrum import (compose_cut, declared_parities,
                          fragment_from_layers)
    code = load_code(args.code)
    layers = load_circuits(args, code)[0]
    n_anc = len(code.checks)
    fr1 = fragment_from_layers(code, layers, 2, "Z", None)
    fr2 = fragment_from_layers(code, layers, 1, None, "Z",
                               anc_init_noise=False)
    det, obs = declared_parities(code, 3, "Z")
    w, sigs, _ = compose_cut(code, fr1, fr2, 2 * n_anc, det, obs)
    ref = get_grader("stim").score(code, layers, 3)[0]
    emit({"code": code.name, "cut": "rounds 1-2 | round 3 + readout",
          "composed_distance": w, "stim_uncut_distance": ref,
          "agree": w == ref,
          "distinct_composite_signatures": len(set(sigs))}, args.out)


def cmd_inject(args):
    from inject import (GOOD_SCHED, frame_circuits, hook_frames,
                        injection_report, standard_frames, surface_pattern,
                        zcheck_hook_config, zcheck_hook_frames)
    code = load_code(args.code)

    init_map = None
    if args.init:
        if len(args.init) != code.n or set(args.init) - {"Z", "X"}:
            sys.exit(f"--init needs a Z/X string of length {code.n}")
        init_map = {q: b for q, b in enumerate(args.init)}

    if not hasattr(code, "_d"):
        # generic CSS path (qLDPC): hook injection from a low-weight
        # Z-logical representative; growth layers must be supplied
        if args.scheme != "hook":
            sys.exit("standard-injection init presets are surface-specific;"
                     " for other codes use --scheme hook")
        if not args.circuit:
            sys.exit("hook injection on a generic code needs --circuit "
                     "FILE with the growth SE schedule (e.g. "
                     "artifacts/bb72_cegis_winners.json)")
        from codes import low_weight_logical_reps
        layers = load_circuits(args, code)[0]
        t0 = time.time()
        reps = low_weight_logical_reps(code, "Z")
        if args.rep >= len(reps):
            sys.exit(f"--rep {args.rep} out of range "
                     f"({len(reps)} representatives found)")
        lz = reps[args.rep]
        try:
            ci, tail, lx = zcheck_hook_config(
                code, lz, tail_size=args.tail_size, lx_seed=args.lx_seed)
            frames = zcheck_hook_frames(code, ci, tail, lz, lx, layers,
                                        args.rounds, init_map=init_map,
                                        r0_checks=args.r0)
            report, results, kept = injection_report(
                code, frames, p=args.p, stim_check=not args.no_stim_check)
        except ValueError as e:
            emit({"code": code.name, "scheme": "hook", "error": str(e)},
                 args.out)
            return
        out = {"code": code.name, "scheme": "hook", "rep": args.rep,
               "lx_seed": args.lx_seed, "r0_checks": args.r0,
               "check": ci, "tail": list(tail), "lz": list(lz),
               "lx": list(lx),
               "se_rounds_after_injection": args.rounds,
               "seconds": round(time.time() - t0, 2)}
        out.update(report)
        if args.crumble:
            out["crumble"] = {
                name: circ.without_noise().to_crumble_url()
                for name, circ in frame_circuits(code, frames, kept, args.p)}
        emit(out, args.out)
        return

    d = code._d
    if args.circuit or args.uniform:
        layers = load_circuits(args, code)[0]
    else:
        layers = layers_from_assignment(
            code, uniform_assignment(code._raw, *GOOD_SCHED), 4)

    t0 = time.time()
    if args.scheme == "standard":
        if not args.site:
            sys.exit("--scheme standard needs --site r,c")
        r, c = (int(x) for x in args.site.split(","))
        site = r * d + c
        im = init_map or surface_pattern(d, r, c, args.pattern)
        frames = standard_frames(code, layers, site, im, args.rounds)
        config = {"scheme": "standard", "site": [r, c],
                  "pattern": None if init_map else args.pattern}
    else:
        frames, pair = hook_frames(code, d, layers, args.rounds,
                                   anchor=args.anchor, init_map=init_map,
                                   head_first=not args.alt_column)
        config = {"scheme": "hook", "anchor": args.anchor,
                  "hook_pair": list(pair)}
    config["init"] = "".join(frames[0]["init_map"].get(q, "Z")
                             for q in range(code.n))
    config["se_rounds_after_injection"] = args.rounds

    try:
        report, results, kept = injection_report(
            code, frames, p=args.p, stim_check=not args.no_stim_check)
    except ValueError as e:
        emit({"code": code.name, **config, "error": str(e)}, args.out)
        return
    out = {"code": code.name, **config,
           "seconds": round(time.time() - t0, 2)}
    out.update(report)
    if args.crumble:
        out["crumble"] = {
            name: circ.without_noise().to_crumble_url()
            for name, circ in frame_circuits(code, frames, kept, args.p)}
    emit(out, args.out)


def cmd_viz(args):
    from viz import TYPES, memory_view, render
    code = load_code(args.code)
    layers = load_circuits(args, code)[0]
    circ = memory_view(code, layers, args.basis, args.rounds, args.noise)
    text = render(circ, args.type)
    if args.type == "crumble":
        print(text)
        return
    if not args.out:
        sys.exit(f"--out FILE.svg required for --type {args.type} "
                 f"(types: {', '.join(TYPES)}, crumble)")
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out}", file=sys.stderr)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        prog="fto", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--code", required=True)
        p.add_argument("--out", help="write JSON here instead of stdout")

    p = sub.add_parser("describe")
    common(p)

    p = sub.add_parser("prep")
    common(p)
    p.add_argument("-T", default="auto")
    p.add_argument("--anc", type=int, default=0)
    p.add_argument("--enum", type=int, default=1)
    p.add_argument("--max-gates", type=int)
    p.add_argument("--ft", action="store_true")
    p.add_argument("--save")

    p = sub.add_parser("synth")
    common(p)
    p.add_argument("-T", default="auto")
    p.add_argument("--library", default="sched",
                   choices=["sched", "clifford"])
    p.add_argument("--tie", action="store_true")
    p.add_argument("--tie-split", type=int,
                   help="refine translation classes by site residue mod P")
    p.add_argument("--hookfree", action="store_true")
    p.add_argument("--max-gates", type=int)
    p.add_argument("--enum", type=int, default=0)
    p.add_argument("--save", help="alias for --out")

    p = sub.add_parser("cegis")
    common(p)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("-T", default="auto")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tie", action="store_true")
    p.add_argument("--tie-split", type=int,
                   help="refine translation classes by site residue mod P")
    p.add_argument("--exhaust", action="store_true")
    p.add_argument("--max-iters", type=int)
    p.add_argument("--full", action="store_true",
                   help="include learned patterns and per-iteration log")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--kernel", choices=["numpy", "gpu"],
                   help="packed min-weight kernel (mim.py) for the scorer")
    p.add_argument("--procs", type=int, default=1,
                   help="workers for the kernel's weight-5 sweep")
    p.add_argument("--save", help="alias for --out")

    for name in ("score", "window", "cut"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--circuit")
        p.add_argument("--uniform")
        if name == "score":
            p.add_argument("--rounds", type=int, default=3)
            p.add_argument("--scorer", default="stim")
            p.add_argument("--explain", action="store_true")
            p.add_argument("--census", action="store_true",
                           help="count accepted-yet-corrupting mechanisms"
                                " and pairs instead of the distance")
            p.add_argument("--cutoff", type=int,
                           help="webs scorer: exact below this weight, "
                                "'>= cutoff' at it (default 4, max 6)")
        if name == "window":
            p.add_argument("--rounds", type=int, default=2)

    p = sub.add_parser("sweep")
    common(p)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--rounds", type=int)
    p.add_argument("--scorer", default="stim")

    p = sub.add_parser("inject")
    common(p)
    p.add_argument("--scheme", required=True, choices=["standard", "hook"])
    p.add_argument("--site", help="data qubit r,c (standard scheme)")
    p.add_argument("--anchor", type=int, default=0,
                   help="top-left data qubit of the 2x2 subpatch (hook)")
    p.add_argument("--alt-column", action="store_true",
                   help="hook the other column of the subpatch")
    p.add_argument("--pattern", default="diag",
                   choices=["diag", "col", "row"])
    p.add_argument("--init", help="explicit Z/X init string (length n)")
    p.add_argument("--rounds", type=int, default=2,
                   help="SE rounds after the injection round")
    p.add_argument("--circuit")
    p.add_argument("--uniform")
    p.add_argument("--p", type=float, default=1e-3)
    p.add_argument("--no-stim-check", action="store_true")
    p.add_argument("--crumble", action="store_true",
                   help="include interactive circuit URLs per frame")
    p.add_argument("--rep", type=int, default=0,
                   help="generic hook: which low-weight Z-logical rep")
    p.add_argument("--lx-seed", type=int, default=0,
                   help="generic hook: X-partner descent seed")
    p.add_argument("--tail-size", type=int,
                   help="generic hook: required |check & logical| overlap")
    p.add_argument("--r0", default="xsafe",
                   choices=["inj", "xsafe", "safe"],
                   help="generic hook: checks measured in the injection "
                        "round besides the hook check")

    p = sub.add_parser("viz")
    common(p)
    p.add_argument("--circuit")
    p.add_argument("--uniform")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--basis", default="Z", choices=["Z", "X"])
    p.add_argument("--type", default="timeline",
                   choices=["timeline", "timeslice", "detslice",
                            "matchgraph", "crumble"])
    p.add_argument("--noise", action="store_true",
                   help="keep noise channels in the diagram")

    args = ap.parse_args()
    {"describe": cmd_describe, "synth": cmd_synth, "score": cmd_score,
     "sweep": cmd_sweep, "window": cmd_window, "cut": cmd_cut,
     "prep": cmd_prep, "cegis": cmd_cegis, "inject": cmd_inject,
     "viz": cmd_viz}[args.cmd](args)


if __name__ == "__main__":
    main()
