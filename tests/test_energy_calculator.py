"""Tests for mcts_crystal.energy_calculator.MaceEnergyCalculator."""

import sys
from types import ModuleType, SimpleNamespace

from ase import Atoms

from mcts_crystal import energy_calculator
from mcts_crystal.energy_calculator import MaceEnergyCalculator


def test_live_calculation_uses_current_thread_calculator(monkeypatch):
    matbench_energy = ModuleType("matbench_discovery.energy")
    matbench_energy.get_e_form_per_atom = lambda payload: payload["energy"]
    monkeypatch.setitem(sys.modules, "matbench_discovery", ModuleType("matbench_discovery"))
    monkeypatch.setitem(sys.modules, "matbench_discovery.energy", matbench_energy)
    monkeypatch.setattr(
        energy_calculator,
        "ExpCellFilter",
        lambda atoms: SimpleNamespace(atoms=atoms),
    )
    monkeypatch.setattr(
        energy_calculator,
        "FIRE",
        lambda atoms_filter: SimpleNamespace(run=lambda fmax: None),
    )
    monkeypatch.setattr(Atoms, "get_total_energy", lambda self: self.calc.energy)
    monkeypatch.setattr(
        MaceEnergyCalculator,
        "_init_calculator",
        lambda self: SimpleNamespace(energy=1.0),
    )

    calc = MaceEnergyCalculator()
    monkeypatch.setattr(calc, "_get_calculator", lambda: SimpleNamespace(energy=7.5))
    monkeypatch.setattr(calc, "_cache_result", lambda *args, **kwargs: None)

    assert calc.calculate_energies(Atoms("H", positions=[[0.0, 0.0, 0.0]])) == (7.5, 7.5)
