"""Visualization: stim diagram rendering of the scored circuits, and the
coordinate layouts shipped with the builtin codes."""

from codes import bb_code, steane_css, surface_css, uniform_assignment
from scoring import circuit_from_gates, layers_from_assignment
from viz import TYPES, memory_view, render

GOOD = (("NW", "SW", "NE", "SE"), ("NW", "NE", "SW", "SE"))


def test_render_all_types_surface():
    code, raw = surface_css(3)
    layers = layers_from_assignment(code, uniform_assignment(raw, *GOOD), 4)
    circ = memory_view(code, layers, "Z", 2)
    for kind in TYPES:
        svg = render(circ, kind)
        assert svg.lstrip().startswith("<svg"), kind
    url = render(circ, "crumble")
    assert url.startswith("https://algassert.com/crumble#circuit=")


def test_builtin_codes_ship_coordinates():
    steane = steane_css()
    assert len(steane.coords) == steane.n + len(steane.checks) == 13
    bb = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert len(bb.coords) == bb.n + len(bb.checks) == 144
    # coordinates reach the stim circuit (timeslice/detslice layouts)
    circ = circuit_from_gates(steane, [[("H", (7,))]], "Z", 1)
    assert "QUBIT_COORDS" in str(circ)
