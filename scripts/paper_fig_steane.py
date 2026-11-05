"""Paper figure generator: the Steane bare-ancilla depth-6 SE winner (R6).

Reads steane_t6_winner.json and emits figures/steane_t6_winner.svg
(standalone vector figure) and figures/steane_t6_winner.tex (quantikz
snippet for the manuscript). Layout: data qubits q0-q6 on top, X-check
ancillas (|+> init, X-basis measurement) then Z-check ancillas (|0>
init, Z-basis measurement) below; six gate layers left to right;
CNOTs colored by the check they realize (X red / Z blue). Within a
layer, gates whose wire spans overlap are offset into sub-columns.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = list(range(7))
XANC = [7, 8, 9]
ZANC = [10, 11, 12]
SUPPORTS = {7: (0, 2, 4, 6), 8: (1, 2, 5, 6), 9: (3, 4, 5, 6),
            10: (0, 2, 4, 6), 11: (1, 2, 5, 6), 12: (3, 4, 5, 6)}
XCOL, ZCOL = "#c0392b", "#2265ae"

with open(ROOT / "artifacts" / "steane_t6_winner.json") as f:
    LAYERS = json.load(f)["circuits"][0]

ROW = {q: i for i, q in enumerate(DATA + XANC + ZANC)}


def wire_y(q):
    r = ROW[q]
    y = 46 + r * 34
    if r >= 7:
        y += 18                      # gap before X ancillas
    if r >= 10:
        y += 18                      # gap before Z ancillas
    return y


def layer_slots(layer):
    """Greedy sub-column assignment so overlapping spans never collide."""
    slots = []                       # slot -> list of occupied (lo, hi)
    out = []
    for _, (a, b) in layer:
        lo, hi = sorted((ROW[a], ROW[b]))
        for si, spans in enumerate(slots):
            if all(hi < s_lo or lo > s_hi for s_lo, s_hi in spans):
                spans.append((lo, hi))
                out.append(si)
                break
        else:
            slots.append([(lo, hi)])
            out.append(len(slots) - 1)
    return out, len(slots)


def svg():
    x = 118
    cols = []                        # (x positions of sub-slots) per layer
    for layer in LAYERS:
        _, n = layer_slots(layer)
        width = 26 * n + 26
        cols.append((x, n))
        x += width
    total_w = x + 150
    total_h = wire_y(12) + 60

    e = [f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'viewBox="0 0 {total_w} {total_h}" font-family="Helvetica">',
         f'<rect width="{total_w}" height="{total_h}" fill="white"/>']

    # wires + left labels + init kets
    for q in DATA + XANC + ZANC:
        y = wire_y(q)
        e.append(f'<line x1="86" y1="{y}" x2="{x + 6}" y2="{y}" '
                 f'stroke="#444" stroke-width="1.3"/>')
        name = (f"q{q}" if q in DATA else f"a{q}")
        e.append(f'<text x="16" y="{y + 4}" font-size="14" '
                 f'fill="#222">{name}</text>')
        if q in XANC:
            ket, col = "|+\u27e9", XCOL
        elif q in ZANC:
            ket, col = "|0\u27e9", ZCOL
        else:
            ket, col = "", ""
        if ket:
            e.append(f'<text x="52" y="{y + 4}" font-size="13" '
                     f'fill="{col}">{ket}</text>')

    # layer headers + gates
    for li, layer in enumerate(LAYERS):
        x0, n = cols[li]
        slots, _ = layer_slots(layer)
        cx_mid = x0 + 13 * (n + 1)
        e.append(f'<text x="{cx_mid}" y="26" font-size="13" fill="#666" '
                 f'text-anchor="middle">{li + 1}</text>')
        for (name, (a, b)), si in zip(layer, slots):
            gx = x0 + 26 * (si + 1)
            anc = a if a >= 7 else b
            col = XCOL if anc in XANC else ZCOL
            ya, yb = wire_y(a), wire_y(b)
            e.append(f'<line x1="{gx}" y1="{ya}" x2="{gx}" y2="{yb}" '
                     f'stroke="{col}" stroke-width="1.6"/>')
            e.append(f'<circle cx="{gx}" cy="{ya}" r="4.2" '
                     f'fill="{col}"/>')                       # control
            e.append(f'<circle cx="{gx}" cy="{yb}" r="8" fill="white" '
                     f'stroke="{col}" stroke-width="1.6"/>')  # target
            e.append(f'<line x1="{gx - 8}" y1="{yb}" x2="{gx + 8}" '
                     f'y2="{yb}" stroke="{col}" stroke-width="1.6"/>')
            e.append(f'<line x1="{gx}" y1="{yb - 8}" x2="{gx}" '
                     f'y2="{yb + 8}" stroke="{col}" stroke-width="1.6"/>')

    # measurement boxes + check labels
    for q in XANC + ZANC:
        y = wire_y(q)
        col = XCOL if q in XANC else ZCOL
        basis = "X" if q in XANC else "Z"
        e.append(f'<rect x="{x + 6}" y="{y - 12}" width="34" height="24" '
                 f'fill="white" stroke="{col}" stroke-width="1.4"/>')
        e.append(f'<text x="{x + 23}" y="{y + 5}" font-size="13" '
                 f'fill="{col}" text-anchor="middle">M{basis}</text>')
        sup = ",".join(str(s) for s in SUPPORTS[q])
        e.append(f'<text x="{x + 48}" y="{y + 4}" font-size="11.5" '
                 f'fill="#777">{basis}({sup})</text>')

    e.append("</svg>")
    return "\n".join(e)


def quantikz():
    order = DATA + XANC + ZANC
    grid = {q: [] for q in order}
    for layer in LAYERS:
        touched = {}
        for _, (a, b) in layer:
            touched[a] = ("c", ROW[b] - ROW[a])
            touched[b] = ("t", None)
        for q in order:
            if q in touched:
                kind, d = touched[q]
                grid[q].append(rf"\ctrl{{{d}}}" if kind == "c"
                               else r"\targ{}")
            else:
                grid[q].append(r"\qw")
    rows = []
    for q in order:
        if q in XANC:
            left, right = r"\lstick{$\ket{+}$}", r"\meter{$X$}"
        elif q in ZANC:
            left, right = r"\lstick{$\ket{0}$}", r"\meter{$Z$}"
        else:
            left, right = rf"\lstick{{$q_{q}$}}", r"\qw"
        rows.append(" & ".join([left] + grid[q] + [right]) + r" \\")
    rows[-1] = rows[-1][:-2].rstrip()
    return ("% Steane [[7,1,3]] bare-ancilla depth-6 SE schedule, full\n"
            "% circuit distance 3 (see docs/REPRODUCIBILITY.md). Generated by\n"
            "% paper_fig_steane.py from steane_t6_winner.json.\n"
            "% Requires \\usepackage{quantikz}\n"
            "\\begin{quantikz}[column sep=0.5em, row sep={0.62cm,between origins}]\n"
            + "\n".join(rows) + "\n\\end{quantikz}\n")


(ROOT / "figures").mkdir(exist_ok=True)
with open(ROOT / "figures/steane_t6_winner.svg", "w") as f:
    f.write(svg())
with open(ROOT / "figures/steane_t6_winner.tex", "w") as f:
    f.write(quantikz())
print("wrote figures/steane_t6_winner.svg and .tex")
