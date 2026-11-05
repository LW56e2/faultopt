"""State preparation: spec generalization beyond SE, end to end."""

import pytest

from codes import steane_css
from scoring import prepares_state
from spectrum import prep_fragment, prep_ft_report
from synth import make_prep_synth


def test_steane_prep_min_depth_is_three():
    code = steane_css()
    assert not make_prep_synth(code, 1).check()
    assert not make_prep_synth(code, 2).check()
    assert make_prep_synth(code, 3).check()


def test_steane_prep_depth3_with_8_cnots():
    """8 CNOTs suffice at the minimal depth 3 (matches the literature's
    best-known gate count while improving on the known depth-4 layouts)."""
    code = steane_css()
    synth = make_prep_synth(code, 3, max_gates=8)
    assert synth.check()
    layers, init = synth.model_layers(), synth.model_init()
    assert sum(len(l) for l in layers) <= 8
    assert prepares_state(code, layers, init)


@pytest.mark.slow
def test_steane_prep_depth3_needs_8_cnots():
    """7 CNOTs at depth 3 is UNSAT (proof): at minimal depth, 8 is the
    CNOT floor. (Whether 7 suffices at greater depth remains open.)"""
    assert not make_prep_synth(steane_css(), 3, max_gates=7).check()


def test_synthesized_circuit_prepares_logical_zero():
    code = steane_css()
    synth = make_prep_synth(code, 3)
    assert synth.check()
    layers = synth.model_layers()
    init = synth.model_init()
    # independent verification via stim tableau simulation
    assert prepares_state(code, layers, init)
    # parallelism cap: 7 qubits -> at most 3 CX per layer
    assert sum(len(l) for l in layers) <= 9


def test_bare_encoding_is_not_fault_tolerant():
    code = steane_css()
    synth = make_prep_synth(code, 3)
    assert synth.check()
    layers, init = synth.model_layers(), synth.model_init()
    rep = prep_ft_report(prep_fragment(7, layers, init, []), 7)
    assert not rep["fault_tolerant_d3"]
    assert len(rep["violations"]) > 0
    # every violation is an irreducible weight->=2 class from one fault
    assert all(v["class_weight"] >= 2 for v in rep["violations"])
