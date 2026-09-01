"""
The ground-state funnel: a structure in, the four Tc descriptors out.

    [vc-relax x2] -> scf -> nscf -> pp.x (ELF cube) -> projwfc.x (PDOS)

Three steps deserve their reasons stated, because skipping them produces
numbers rather than errors:

**Relaxing at pressure takes two passes.** The plane-wave basis is defined on
the cell the run *started* from, so as vc-relax changes the cell the stress it
is steering by carries a Pulay error. Pass 1 can declare itself converged at
0.2 kbar and be 100+ kbar out when the stress is recomputed on that cell. Run
vc-relax, take the final cell, run vc-relax again from it, and read the *Final
scf* pressure rather than the one BFGS quotes.

**The SCF must be fresh on the relaxed cell.** Dumping the ELF from the last
step of a variable-cell relaxation gives the field of the previous basis.

**The descriptors are not comparable across protocols.** phi and phi* are
percolation thresholds on the SCF's FFT grid, whose resolution moves with the
plane-wave cutoff. Two networking values from different cutoffs, meshes or
pseudopotential families are not a comparison.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from ..descriptors import DEFAULT_TOL, compute_descriptors, read_elf_cube
from .inputs import (
    QESettings,
    hydrogen_fractional_coordinates,
    write_pp_elf_input,
    write_projwfc_input,
    write_pw_input,
)
from .outputs import hydrogen_dos_fraction, parse_pw_output, relaxed_atoms
from .runner import QEError, QERunner

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

PREFIX = "sh"
OUTDIR = "./scratch"
ELF_CUBE = "elf.cube"


@dataclass(frozen=True)
class GroundStateResult:
    """Everything the Tc fit needs, plus what it took to get there."""

    phi: float
    phi_star: float
    h_f: float
    h_dos: float
    fermi_ev: float
    pressure_gpa: Optional[float]
    energy_ry: Optional[float]
    grid_shape: tuple
    #: The relaxed structure, when the funnel relaxed; otherwise the input one.
    atoms: Optional["Atoms"] = None

    def as_properties(self) -> Dict[str, float]:
        """The descriptor dict a RewardFunction consumes."""
        return {
            "phi": self.phi,
            "phi_star": self.phi_star,
            "h_f": self.h_f,
            "h_dos": self.h_dos,
        }


def run_ground_state(
    atoms: "Atoms",
    settings: QESettings,
    runner: QERunner,
    workdir: str,
    *,
    pressure_gpa: Optional[float] = None,
    relax: bool = True,
    relax_passes: int = 2,
    keep_cube: bool = False,
    tol: float = DEFAULT_TOL,
) -> GroundStateResult:
    """
    Run the funnel and return the four descriptors.

    Args:
        atoms: the candidate structure.
        settings: the numerical protocol, held fixed across a campaign.
        runner: how to invoke the QE binaries.
        workdir: this candidate's private directory. One run, one directory.
        pressure_gpa: target pressure for the relaxation. Required when
            ``relax`` is True - a hydride relaxed to 0 GPa is a different
            material from the same hydride at 200 GPa.
        relax: run vc-relax before the SCF. Set False when the structure is
            already relaxed at this pressure with this protocol.
        relax_passes: vc-relax passes. 2 is the minimum that removes the Pulay
            error; if the second pass still moves the cell appreciably, raise it
            (and suspect the cutoff is too low for the stress).
        keep_cube: keep the ELF cube on disk. Cubes are tens of megabytes each;
            a campaign that keeps every one fills a filesystem long before it
            finishes.
        tol: bisection tolerance on the ELF threshold.

    Returns:
        A :class:`GroundStateResult`.

    Raises:
        ValueError: on relax without a pressure, or a structure with no hydrogen.
        QEError: if any step fails to reach 'JOB DONE.', the geometry
            optimisation does not converge, or the ELF cube is not written.
    """
    if relax and pressure_gpa is None:
        raise ValueError(
            "run_ground_state(relax=True) needs pressure_gpa. Relaxing a "
            "pressure-stabilised hydride to 0 GPa gives a different material."
        )
    if "H" not in atoms.get_chemical_symbols():
        raise ValueError("Structure contains no hydrogen; it is not a hydride")

    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    ranks = runner.ranks_for(atoms, settings.ecutwfc)
    current = atoms.copy()

    # 1. Relax at the target pressure, twice, to shed the Pulay error.
    if relax:
        for pass_index in range(1, relax_passes + 1):
            stem = f"vcrelax{pass_index}"
            text = runner.run(
                "pw.x",
                write_pw_input(
                    current,
                    settings,
                    calculation="vc-relax",
                    prefix=PREFIX,
                    outdir=OUTDIR,
                    pressure_gpa=pressure_gpa,
                ),
                workdir,
                stem=stem,
                ranks=ranks,
            )
            result = parse_pw_output(text)
            if not result.ok(require_relaxed=True):
                raise QEError(
                    f"vc-relax pass {pass_index} in {workdir}: "
                    f"{result.failure_reason(require_relaxed=True)}"
                )
            current = relaxed_atoms(result, current)
            logger.info(
                "vc-relax pass %d: final scf P = %s GPa",
                pass_index,
                None if result.pressure_gpa is None else round(result.pressure_gpa, 2),
            )
            # The cell changed, so the plane count did too.
            ranks = runner.ranks_for(current, settings.ecutwfc)

    # 2. A fresh SCF on the cell whose ELF we are about to dump.
    scf_text = runner.run(
        "pw.x",
        write_pw_input(
            current, settings, calculation="scf", prefix=PREFIX, outdir=OUTDIR
        ),
        workdir,
        stem="scf",
        ranks=ranks,
    )
    scf = parse_pw_output(scf_text)
    if not scf.ok():
        raise QEError(f"scf in {workdir}: {scf.failure_reason()}")

    # 3. NSCF on a denser mesh - what the ELF and the projected DOS are read from.
    nscf_text = runner.run(
        "pw.x",
        write_pw_input(
            current, settings, calculation="nscf", prefix=PREFIX, outdir=OUTDIR
        ),
        workdir,
        stem="nscf",
        ranks=ranks,
    )
    nscf = parse_pw_output(nscf_text)
    if not nscf.ok():
        raise QEError(f"nscf in {workdir}: {nscf.failure_reason()}")

    fermi_ev = nscf.fermi_ev if nscf.fermi_ev is not None else scf.fermi_ev
    if fermi_ev is None:
        raise QEError(
            f"No Fermi level in {workdir}: without it the hydrogen DOS share is "
            f"undefined. A hydride under pressure should be a metal - check the "
            f"occupations."
        )

    # 4. The ELF cube.
    runner.run(
        "pp.x",
        write_pp_elf_input(
            prefix=PREFIX, outdir=OUTDIR, filplot="elf.dat", fileout=ELF_CUBE
        ),
        workdir,
        stem="pp_elf",
        ranks=ranks,
    )
    cube_path = directory / ELF_CUBE
    if not cube_path.exists():
        raise QEError(f"pp.x finished but wrote no {ELF_CUBE} in {workdir}")

    # 5. The atom-projected DOS.
    runner.run(
        "projwfc.x",
        write_projwfc_input(prefix=PREFIX, outdir=OUTDIR, degauss=settings.degauss),
        workdir,
        stem="projwfc",
        ranks=ranks,
    )
    # projwfc.x resolves filpdos against its working directory, not outdir.
    h_dos = hydrogen_dos_fraction(str(directory), PREFIX, fermi_ev)

    # 6. The percolation descriptors. Hydrogen positions come from the
    #    structure - QE writes the species index where a cube declares Z.
    elf, _origin, _cell, _cube_atoms = read_elf_cube(str(cube_path))
    descriptors = compute_descriptors(
        elf,
        hydrogen_fractional_coordinates(current),
        current.get_atomic_numbers(),
        tol=tol,
    )

    if not keep_cube:
        cube_path.unlink(missing_ok=True)
        (directory / "elf.dat").unlink(missing_ok=True)

    return GroundStateResult(
        phi=descriptors.phi,
        phi_star=descriptors.phi_star,
        h_f=descriptors.h_f,
        h_dos=h_dos,
        fermi_ev=fermi_ev,
        pressure_gpa=scf.pressure_gpa,
        energy_ry=scf.energy_ry,
        grid_shape=descriptors.grid_shape,
        atoms=current,
    )


def clean_scratch(workdir: str) -> None:
    """
    Delete a candidate's QE scratch directory, keeping the inputs and outputs.

    Wavefunctions dominate the footprint and nothing downstream reads them once
    the descriptors are out.
    """
    shutil.rmtree(Path(workdir) / OUTDIR, ignore_errors=True)
