"""
Quantum ESPRESSO ground-state calculations for the superhydride descriptors.

The funnel is:

    [vc-relax x2] -> scf -> nscf -> pp.x (ELF cube) -> projwfc.x (PDOS)

which yields phi and phi* from the ELF, H_DOS from the projection, and H_f from
the composition - the four inputs to the Tc fit.

    QESettings                 the numerical protocol
    QERunner                   how to invoke pw.x / pp.x / projwfc.x
    run_ground_state           the funnel
    QuantumEspressoEvaluator   the funnel as a framework PropertyEvaluator

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from .evaluator import QuantumEspressoEvaluator
from .inputs import (
    QESettings,
    hydrogen_fractional_coordinates,
    kgrid_from_spacing,
    write_pp_elf_input,
    write_projwfc_input,
    write_pw_input,
)
from .outputs import PwResult, hydrogen_dos_fraction, parse_pw_output, relaxed_atoms
from .pipeline import GroundStateResult, clean_scratch, run_ground_state
from .runner import QEError, QERunner

__all__ = [
    "QESettings",
    "QERunner",
    "QEError",
    "PwResult",
    "GroundStateResult",
    "QuantumEspressoEvaluator",
    "run_ground_state",
    "clean_scratch",
    "write_pw_input",
    "write_pp_elf_input",
    "write_projwfc_input",
    "kgrid_from_spacing",
    "hydrogen_fractional_coordinates",
    "parse_pw_output",
    "relaxed_atoms",
    "hydrogen_dos_fraction",
]
