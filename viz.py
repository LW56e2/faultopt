"""TOOL: circuit visualization through stim's built-in diagram renderers.

Renders the EXACT circuits the scorers score (built by the same
builders), so what you look at is what was graded:

  timeline    gate-by-gate SVG timeline
  timeslice   per-tick 2D layout (uses QUBIT_COORDS; the surface, Steane
              and BB builtins all ship coordinates)
  detslice    detector-slice SVG -- the shaded detecting regions are the
              Pauli webs of the closed circuit, drawn by stim
  matchgraph  the decoding graph
  crumble     interactive-editor URL (algassert.com/crumble): step
              through ticks, toggle detecting regions live

SVG renderers accept noisy circuits; crumble URLs strip noise for
readability (the gate structure is unchanged).
"""

from scoring import circuit_from_gates

TYPES = {"timeline": "timeline-svg", "timeslice": "timeslice-svg",
         "detslice": "detslice-with-ops-svg", "matchgraph": "match-graph-svg"}


def memory_view(code, layers, basis="Z", rounds=2, noise=False, p=1e-3):
    """The memory-experiment circuit for gate layers, as scored."""
    c = circuit_from_gates(code, layers, basis, rounds, p)
    return c if noise else c.without_noise()


def render(circ, kind):
    """SVG text for the diagram kinds, or the crumble URL."""
    if kind == "crumble":
        return circ.without_noise().to_crumble_url()
    return str(circ.diagram(TYPES[kind]))
