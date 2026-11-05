"""Poster figure: the SAT encoding as a grid of placeholder gates.

Five generic wires x five layers. Candidate CNOTs are dashed gray
ghosts (selector off), chosen gates are solid blue (selector on), one
slot is amber with "?" (solver deciding), and one ghost is marked
blocked to illustrate the at-most-one-gate-per-qubit-per-layer rule.
Emits figures/sat_grid.svg.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

W, H = 940, 500
NW, NL = 5, 5
X0, XSTEP = 150, 150
Y0, YSTEP = 92, 62
BLUE, GRAY, AMBER, RED = "#2265ae", "#9aa2ab", "#d9871c", "#c0392b"

def wy(i):
    return Y0 + i * YSTEP

def cx(col):
    return X0 + col * XSTEP + XSTEP // 2

e = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     f'font-family="Helvetica">',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# column lanes + headers
for c in range(NL):
    x = X0 + c * XSTEP
    e.append(f'<rect x="{x + 14}" y="{Y0 - 34}" width="{XSTEP - 28}" '
             f'height="{(NW - 1) * YSTEP + 62}" fill="#f4f6f8" rx="10"/>')
    lab = f"layer {c + 1}" if c < NL - 1 else "layer T"
    e.append(f'<text x="{cx(c)}" y="{Y0 - 46}" font-size="15" fill="#555" '
             f'text-anchor="middle">{lab}</text>')

# wires
for i in range(NW):
    y = wy(i)
    e.append(f'<line x1="{X0 - 44}" y1="{y}" x2="{X0 + NL * XSTEP + 10}" '
             f'y2="{y}" stroke="#333" stroke-width="1.4"/>')
    e.append(f'<text x="{X0 - 74}" y="{y + 5}" font-size="15" '
             f'fill="#222">q{i + 1}</text>')

def cnot(col, ctrl, targ, color, dashed=False, slot_dx=0, qmark=False):
    x = cx(col) + slot_dx
    y1, y2 = wy(ctrl), wy(targ)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    e.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
             f'stroke="{color}" stroke-width="2"{dash}/>')
    e.append(f'<circle cx="{x}" cy="{y1}" r="5" fill="{color if not dashed else "white"}" '
             f'stroke="{color}" stroke-width="2"{dash}/>')
    e.append(f'<circle cx="{x}" cy="{y2}" r="9" fill="white" '
             f'stroke="{color}" stroke-width="2"{dash}/>')
    e.append(f'<line x1="{x - 9}" y1="{y2}" x2="{x + 9}" y2="{y2}" '
             f'stroke="{color}" stroke-width="2"{dash}/>')
    e.append(f'<line x1="{x}" y1="{y2 - 9}" x2="{x}" y2="{y2 + 9}" '
             f'stroke="{color}" stroke-width="2"{dash}/>')
    if qmark:
        e.append(f'<text x="{x + 15}" y="{(y1 + y2) // 2 + 5}" '
                 f'font-size="17" font-weight="700" fill="{color}">?</text>')

# layer 1: one chosen gate + one ghost
cnot(0, 0, 2, BLUE, slot_dx=-22)
cnot(0, 3, 4, GRAY, dashed=True, slot_dx=22)
# layer 2: chosen gate + a BLOCKED ghost sharing wire q2 (index 1)
cnot(1, 1, 3, BLUE, slot_dx=-22)
cnot(1, 0, 1, GRAY, dashed=True, slot_dx=24)
bx, by = cx(1) + 24, (wy(0) + wy(1)) // 2
e.append(f'<line x1="{bx - 9}" y1="{by - 9}" x2="{bx + 9}" y2="{by + 9}" '
         f'stroke="{RED}" stroke-width="2.4"/>')
e.append(f'<line x1="{bx - 9}" y1="{by + 9}" x2="{bx + 9}" y2="{by - 9}" '
         f'stroke="{RED}" stroke-width="2.4"/>')
# layer 3: undecided slot
cnot(2, 1, 4, AMBER, dashed=True, qmark=True)
# layer 4: two parallel chosen gates (disjoint wires)
cnot(3, 0, 1, BLUE, slot_dx=-22)
cnot(3, 2, 3, BLUE, slot_dx=22)
# layer T: ghosts only
cnot(4, 0, 3, GRAY, dashed=True, slot_dx=-22)
cnot(4, 2, 4, GRAY, dashed=True, slot_dx=22)

# legend, two rows so nothing overflows the canvas
items = [(BLUE, False, "gate selected  (variable = true)"),
         (GRAY, True, "candidate, not selected  (variable = false)"),
         (AMBER, True, "solver deciding"),
         (RED, None, "forbidden: wire already busy this layer")]
for k, (col, dashed, text) in enumerate(items):
    lx = X0 - 44 + (k % 2) * 370
    ly = wy(NW - 1) + 54 + (k // 2) * 28
    if dashed is None:
        e.append(f'<line x1="{lx}" y1="{ly - 5}" x2="{lx + 14}" '
                 f'y2="{ly + 9}" stroke="{col}" stroke-width="2.4"/>')
        e.append(f'<line x1="{lx}" y1="{ly + 9}" x2="{lx + 14}" '
                 f'y2="{ly - 5}" stroke="{col}" stroke-width="2.4"/>')
    else:
        d = ' stroke-dasharray="5 4"' if dashed else ""
        e.append(f'<line x1="{lx}" y1="{ly + 2}" x2="{lx + 16}" '
                 f'y2="{ly + 2}" stroke="{col}" stroke-width="2.6"{d}/>')
    e.append(f'<text x="{lx + 24}" y="{ly + 7}" font-size="13.5" '
             f'fill="#333">{text}</text>')

e.append("</svg>")
(ROOT / "figures").mkdir(exist_ok=True)
with open(ROOT / "figures/sat_grid.svg", "w") as f:
    f.write("\n".join(e))
print("wrote figures/sat_grid.svg")
