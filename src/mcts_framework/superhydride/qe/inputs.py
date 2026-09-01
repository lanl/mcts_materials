"""
Quantum ESPRESSO input generation for the superhydride descriptor funnel.

Hand-rolled rather than delegated to ASE's or pymatgen's writers, because the
funnel needs ``pp.x`` ELF cubes and ``projwfc.x`` projections that neither
emits, and because these inputs *are* the scientific protocol - they should be
readable in one file.

Units follow QE: energies in rydberg, pressure in kbar, lengths in angstrom via
``CELL_PARAMETERS``.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid importing ase at module load time
    from ase import Atoms

GPA_TO_KBAR = 10.0

#: ELF is plot_num = 8 in pp.x.
PP_PLOT_NUM_ELF = 8
#: Gaussian cube is output_format = 6, iflag = 3 (3D).
PP_OUTPUT_FORMAT_CUBE = 6


@dataclass
class QESettings:
    """
    The numerical protocol. One instance per campaign, held fixed across every
    structure in it - two enthalpies or two Fermi-level quantities computed at
    different ``degauss`` or different cutoffs are not comparable, and neither
    are two networking values.

    Defaults follow Belli et al. (Ann. Phys. 2025, 537, e00280) where the
    choice is physical, and diverge where the pseudopotential type makes their
    choice wrong for a different one - see ``ecutrho``.
    """

    #: Plane-wave cutoff (Ry). The paper used 90 Ry. This is the real cost knob
    #: and must be converged for the property you report.
    ecutwfc: float = 90.0

    #: Charge-density cutoff (Ry). 4x ecutwfc, which is exact for
    #: NORM-CONSERVING pseudopotentials: the density then has no Fourier
    #: components beyond 4x, and nothing above that ratio changes any printed
    #: digit. Ultrasoft and PAW need far more - the paper used 900 Ry with PAW,
    #: a ratio of 10. Check ``pseudo_type`` in the UPF header before trusting
    #: this default.
    ecutrho: float = 360.0

    #: A hydride under pressure is a metal: fixed occupations would converge to
    #: a wrong ground state and only complain if the charge is off by >0.1%.
    occupations: str = "smearing"
    #: Methfessel-Paxton: its free-energy error is second order in degauss, so
    #: the T->0 energy barely depends on the value. Gaussian is first order.
    smearing: str = "mp"
    #: Ry. Not a free parameter - it trades against the k-mesh, so the two are
    #: converged together and then held fixed. The paper used 0.02 Ry.
    degauss: float = 0.02

    #: Ry. Everything downstream is a derivative of an energy converged to this,
    #: so they inherit it amplified. Never loosen it to get a stuck run past a
    #: failure - that turns a visible failure into an invisible one.
    conv_thr: float = 1e-10
    forc_conv_thr: float = 1e-4      # Ry/bohr
    press_conv_thr: float = 0.5      # kbar
    nstep: int = 200

    #: k-point spacing (1/angstrom) for the SCF, as |b_i| / n_i where the
    #: reciprocal vectors carry the 2*pi. The paper's 2*pi x 0.036 1/A.
    #: A spacing, not a grid: a fixed n x n x n is coarse in a small cell and
    #: wasteful in a large one, and in an elongated cell it is both at once.
    kspacing_scf: float = 2.0 * math.pi * 0.036
    #: Denser mesh for the NSCF that feeds the ELF and the projected DOS.
    kspacing_nscf: float = 2.0 * math.pi * 0.018

    #: Directory holding the UPF files. Never write a pseudopotential filename
    #: you have not seen on disk, and never mix families within a set of numbers
    #: meant to be compared.
    pseudo_dir: str = ""
    #: Explicit element -> UPF filename map. Left empty, the writer looks for
    #: "<Element>.upf" in pseudo_dir.
    pseudo_files: Dict[str, str] = field(default_factory=dict)

    def ecut_ratio(self) -> float:
        """ecutrho / ecutwfc, the number to sanity-check against the UPF type."""
        return self.ecutrho / self.ecutwfc


# --- k-point meshes -------------------------------------------------------


def kgrid_from_spacing(cell: np.ndarray, spacing: float) -> Tuple[int, int, int]:
    """
    Monkhorst-Pack grid sampling the Brillouin zone at roughly ``spacing``.

    Args:
        cell: 3x3 real-space lattice matrix, rows are the lattice vectors, in
            angstrom.
        spacing: target |b_i| / n_i in 1/angstrom, with the 2*pi convention.

    Returns:
        ``(n1, n2, n3)``, each at least 1.

    Points are distributed per reciprocal lattice vector, so an elongated cell
    gets fewer along its long real-space axis - which is what you want, and
    what a fixed cube gets wrong.
    """
    if spacing <= 0:
        raise ValueError(f"k-point spacing must be positive, got {spacing}")
    reciprocal = 2.0 * math.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
    lengths = np.linalg.norm(reciprocal, axis=1)
    grid = np.maximum(np.ceil(lengths / spacing).astype(int), 1)
    return int(grid[0]), int(grid[1]), int(grid[2])


# --- pw.x -----------------------------------------------------------------


def pseudo_filename(element: str, settings: QESettings) -> str:
    """UPF filename for an element, from the explicit map or the default pattern."""
    return settings.pseudo_files.get(element, f"{element}.upf")


def _species(atoms: "Atoms") -> List[str]:
    """Chemical symbols present, in a stable order."""
    return sorted(set(atoms.get_chemical_symbols()))


def _cards(atoms: "Atoms", settings: QESettings, kgrid: Tuple[int, int, int]) -> str:
    """The ATOMIC_SPECIES / ATOMIC_POSITIONS / CELL_PARAMETERS / K_POINTS block."""
    from ase.data import atomic_masses, atomic_numbers

    lines = ["ATOMIC_SPECIES"]
    for element in _species(atoms):
        mass = atomic_masses[atomic_numbers[element]]
        lines.append(f"  {element}  {mass:.4f}  {pseudo_filename(element, settings)}")

    lines += ["", "ATOMIC_POSITIONS crystal"]
    for symbol, (a, b, c) in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
        lines.append(f"  {symbol}  {a:.10f}  {b:.10f}  {c:.10f}")

    lines += ["", "CELL_PARAMETERS angstrom"]
    for row in np.asarray(atoms.get_cell()):
        lines.append(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")

    lines += ["", "K_POINTS automatic", f"  {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0"]
    return "\n".join(lines)


def write_pw_input(
    atoms: "Atoms",
    settings: QESettings,
    *,
    calculation: str,
    prefix: str,
    outdir: str,
    pressure_gpa: Optional[float] = None,
    kgrid: Optional[Tuple[int, int, int]] = None,
) -> str:
    """
    Build a pw.x input deck.

    Args:
        atoms: the structure.
        settings: the numerical protocol.
        calculation: 'scf', 'nscf', 'relax' or 'vc-relax'.
        prefix, outdir: QE's prefix and scratch directory. Give every task its
            own working directory AND its own outdir - concurrent runs sharing
            a scratch directory read each other's wavefunctions and return
            plausible numbers belonging to a different structure, with no error.
        pressure_gpa: target pressure. REQUIRED for vc-relax: a variable-cell
            relaxation with no target silently relaxes to zero pressure, which
            for a hydride is a different material.
        kgrid: explicit mesh; defaults to the spacing for this calculation type.

    With ``occupations = 'smearing'`` QE sizes the band count itself, adding
    empty states above E_F, so the projected DOS has something to project onto
    without an explicit ``nbnd``.

    Raises:
        ValueError: on an unknown calculation, a vc-relax without a pressure,
            or an empty pseudo_dir.
    """
    if calculation not in ("scf", "nscf", "relax", "vc-relax"):
        raise ValueError(f"Unsupported pw.x calculation: {calculation!r}")
    if calculation == "vc-relax" and pressure_gpa is None:
        raise ValueError(
            "vc-relax without a target pressure relaxes to 0 GPa. Pass "
            "pressure_gpa explicitly - for a pressure-stabilised hydride the "
            "ambient-pressure cell is a different material."
        )
    if not settings.pseudo_dir:
        raise ValueError("QESettings.pseudo_dir is empty; pw.x needs a pseudopotential directory")

    if kgrid is None:
        spacing = (
            settings.kspacing_nscf if calculation == "nscf" else settings.kspacing_scf
        )
        kgrid = kgrid_from_spacing(np.asarray(atoms.get_cell()), spacing)

    control = [
        "&CONTROL",
        f"  calculation = '{calculation}'",
        f"  prefix = '{prefix}'",
        f"  outdir = '{outdir}'",
        f"  pseudo_dir = '{settings.pseudo_dir}'",
        "  tprnfor = .true.",
        "  tstress = .true.",
    ]
    if calculation in ("relax", "vc-relax"):
        control.append(f"  forc_conv_thr = {settings.forc_conv_thr}")
        control.append(f"  nstep = {settings.nstep}")
    control.append("/")

    system = [
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {len(atoms)}",
        f"  ntyp = {len(_species(atoms))}",
        f"  ecutwfc = {settings.ecutwfc}",
        f"  ecutrho = {settings.ecutrho}",
        f"  occupations = '{settings.occupations}'",
        f"  smearing = '{settings.smearing}'",
        f"  degauss = {settings.degauss}",
    ]
    system.append("/")

    electrons = ["&ELECTRONS", f"  conv_thr = {settings.conv_thr}", "/"]

    blocks = [control, system, electrons]
    if calculation in ("relax", "vc-relax"):
        blocks.append(["&IONS", "  ion_dynamics = 'bfgs'", "/"])
    if calculation == "vc-relax":
        blocks.append([
            "&CELL",
            "  cell_dynamics = 'bfgs'",
            f"  press = {pressure_gpa * GPA_TO_KBAR}",
            f"  press_conv_thr = {settings.press_conv_thr}",
            "/",
        ])

    namelists = "\n".join("\n".join(block) for block in blocks)
    return f"{namelists}\n\n{_cards(atoms, settings, kgrid)}\n"



# --- pp.x and projwfc.x ---------------------------------------------------


def write_pp_elf_input(
    *, prefix: str, outdir: str, filplot: str, fileout: str
) -> str:
    """
    Build a pp.x input that dumps the ELF as a Gaussian cube.

    The grid is the SCF's FFT grid, not a choice: a ten-atom cell at 80 Ry is
    roughly 45^3. Both descriptors are percolation thresholds on that grid, so
    its resolution floors their precision - and it moves with the cutoff, which
    is why descriptors from different cutoffs are not comparable.
    """
    return (
        "&INPUTPP\n"
        f"  prefix = '{prefix}'\n"
        f"  outdir = '{outdir}'\n"
        f"  plot_num = {PP_PLOT_NUM_ELF}\n"
        f"  filplot = '{filplot}'\n"
        "/\n"
        "&PLOT\n"
        "  iflag = 3\n"
        f"  output_format = {PP_OUTPUT_FORMAT_CUBE}\n"
        f"  fileout = '{fileout}'\n"
        "/\n"
    )


def write_projwfc_input(
    *,
    prefix: str,
    outdir: str,
    degauss: float,
    filpdos: Optional[str] = None,
    delta_e: float = 0.02,
    ngauss: int = 0,
) -> str:
    """
    Build a projwfc.x input for the atom-projected DOS.

    H_DOS - the hydrogen share of the DOS at E_F - is read off these files.
    It is a projection onto atomic orbitals, so it is sensitive to the radii
    the pseudopotentials imply; the paper flags exactly that as a likely source
    of the scatter in the H_DOS/Tc trend.

    Args:
        filpdos: prefix for the per-orbital PDOS files. projwfc.x resolves it
            against its WORKING directory, not against outdir - set it
            explicitly so the reader knows where to look. Defaults to `prefix`.
        delta_e: energy step (eV) of the tabulated DOS.
        ngauss: broadening type; 0 = simple Gaussian, which is what a DOS
            wants regardless of the smearing used to occupy the SCF.
    """
    return (
        "&PROJWFC\n"
        f"  prefix = '{prefix}'\n"
        f"  outdir = '{outdir}'\n"
        f"  filpdos = '{filpdos or prefix}'\n"
        f"  ngauss = {ngauss}\n"
        f"  degauss = {degauss}\n"
        f"  DeltaE = {delta_e}\n"
        "/\n"
    )


def hydrogen_fractional_coordinates(atoms: "Atoms") -> np.ndarray:
    """
    Fractional coordinates of the hydrogen atoms, for the molecularity index.

    Taken from the structure rather than from the cube's atom block: QE writes
    the *species index* in the column a Gaussian cube reserves for the atomic
    number, so a filter on "Z == 1" there finds hydrogen only by luck.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    return np.asarray(atoms.get_scaled_positions())[symbols == "H"]

