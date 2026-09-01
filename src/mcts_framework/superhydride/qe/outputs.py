"""
Quantum ESPRESSO output parsing.

Two rules run through everything here:

    * **``JOB DONE.`` is the gate, not the exit code.** pw.x exits 0 on several
      genuine failures, ``convergence NOT achieved`` among them, and a relax
      that exhausts ``nstep`` prints ``JOB DONE.`` while writing no final
      coordinates at all.
    * **From a relax, quote the LAST values in the file.** After ``Final scf
      calculation at the relaxed structure`` those are the numbers on the
      relaxed cell in a basis that fits it; the ones BFGS was steering by carry
      a Pulay error and can be off by 100 kbar.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

KBAR_TO_GPA = 0.1

_JOB_DONE = "JOB DONE."
_ENERGY = re.compile(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", re.MULTILINE)
_PRESSURE = re.compile(r"P=\s*(-?[\d.]+)")
_FERMI = re.compile(r"the Fermi energy is\s+(-?[\d.]+)\s*ev")
_HIGHEST_OCCUPIED = re.compile(r"highest occupied.*?level \(ev\):\s+(-?[\d.]+)")
_BFGS_CONVERGED = "End of BFGS Geometry Optimization"
_NOT_CONVERGED = "convergence NOT achieved"
_BFGS_FAILED = "bfgs failed"


@dataclass
class PwResult:
    """What a pw.x run says about the structure it was given."""

    job_done: bool
    energy_ry: Optional[float] = None
    pressure_kbar: Optional[float] = None
    fermi_ev: Optional[float] = None
    #: For relax/vc-relax: the relaxed cell (angstrom) and fractional positions.
    final_cell: Optional[np.ndarray] = None
    final_scaled_positions: Optional[np.ndarray] = None
    final_symbols: Optional[List[str]] = None
    #: None for a single-point run; True/False for a geometry optimisation.
    bfgs_converged: Optional[bool] = None
    scf_converged: bool = True

    @property
    def pressure_gpa(self) -> Optional[float]:
        if self.pressure_kbar is None:
            return None
        return self.pressure_kbar * KBAR_TO_GPA

    def ok(self, *, require_relaxed: bool = False) -> bool:
        """
        True if this run may be quoted from.

        Args:
            require_relaxed: also demand a converged geometry optimisation with
                a final-coordinates block.
        """
        if not (self.job_done and self.scf_converged):
            return False
        if require_relaxed:
            return bool(self.bfgs_converged) and self.final_cell is not None
        return True

    def failure_reason(self, *, require_relaxed: bool = False) -> str:
        """A short description of why :meth:`ok` is False (empty string if it is True)."""
        if not self.job_done:
            return "pw.x did not print 'JOB DONE.'"
        if not self.scf_converged:
            return "SCF convergence not achieved"
        if require_relaxed and not self.bfgs_converged:
            return "geometry optimisation did not converge (nstep exhausted or bfgs failed)"
        if require_relaxed and self.final_cell is None:
            return "no final-coordinates block was written"
        return ""


def parse_pw_output(text: str) -> PwResult:
    """
    Parse a pw.x stdout capture.

    Energy, pressure and Fermi level are taken from the LAST occurrence in the
    file, which for a relax is the final SCF on the relaxed cell.
    """
    job_done = _JOB_DONE in text
    scf_converged = _NOT_CONVERGED not in text

    energies = _ENERGY.findall(text)
    pressures = _PRESSURE.findall(text)
    fermis = _FERMI.findall(text) or _HIGHEST_OCCUPIED.findall(text)

    bfgs_converged: Optional[bool] = None
    if "Begin final coordinates" in text or "BFGS Geometry Optimization" in text:
        bfgs_converged = _BFGS_CONVERGED in text and _BFGS_FAILED not in text

    cell, positions, symbols = _parse_final_coordinates(text)

    return PwResult(
        job_done=job_done,
        energy_ry=float(energies[-1]) if energies else None,
        pressure_kbar=float(pressures[-1]) if pressures else None,
        fermi_ev=float(fermis[-1]) if fermis else None,
        final_cell=cell,
        final_scaled_positions=positions,
        final_symbols=symbols,
        bfgs_converged=bfgs_converged,
        scf_converged=scf_converged,
    )


def _parse_final_coordinates(
    text: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Pull the cell and fractional positions out of the final-coordinates block."""
    start = text.rfind("Begin final coordinates")
    if start < 0:
        return None, None, None
    end = text.find("End final coordinates", start)
    block = text[start : end if end > 0 else len(text)]

    cell = None
    match = re.search(r"CELL_PARAMETERS \((angstrom|bohr|alat=?\s*[\d.]*)\)(.*?)\n\s*\n",
                      block, re.DOTALL)
    if match:
        rows = [
            [float(x) for x in line.split()[:3]]
            for line in match.group(2).strip().splitlines()
            if len(line.split()) >= 3
        ]
        if len(rows) == 3:
            cell = np.array(rows)
            if match.group(1) == "bohr":
                cell *= 0.529177210903

    symbols: List[str] = []
    fractional: List[List[float]] = []
    match = re.search(r"ATOMIC_POSITIONS \((crystal|angstrom|bohr|alat)\)(.*)",
                      block, re.DOTALL)
    if match and match.group(1) == "crystal":
        for line in match.group(2).strip().splitlines():
            parts = line.split()
            if len(parts) >= 4 and re.fullmatch(r"[A-Z][a-z]?\d*", parts[0]):
                symbols.append(re.sub(r"\d+$", "", parts[0]))
                fractional.append([float(x) for x in parts[1:4]])

    positions = np.array(fractional) if fractional else None
    return cell, positions, (symbols or None)


def relaxed_atoms(result: PwResult, template: "object") -> "object":
    """
    Return a copy of ``template`` (an ASE Atoms) carrying the relaxed geometry.

    Raises:
        ValueError: if the result has no final-coordinates block.
    """
    if result.final_cell is None or result.final_scaled_positions is None:
        raise ValueError("This pw.x result carries no relaxed structure")
    atoms = template.copy()
    atoms.set_cell(result.final_cell)
    atoms.set_scaled_positions(result.final_scaled_positions)
    return atoms


# --- projwfc.x ------------------------------------------------------------

#: projwfc.x names its per-orbital files "<prefix>.pdos_atm#<n>(<El>)_wfc#<m>(<l>)".
_PDOS_ATOM_FILE = re.compile(r"\.pdos_atm#(\d+)\(([A-Za-z]{1,2})\)_wfc#(\d+)\(([a-z])\)$")


def hydrogen_dos_fraction(
    outdir: str, prefix: str, fermi_ev: float, *, element: str = "H"
) -> float:
    """
    Hydrogen's share of the projected DOS at the Fermi level.

    Both numerator and denominator are sums over the same per-orbital projection
    files, so the result is a genuine share in [0, 1] and does not depend on how
    much DOS the projection fails to account for.

    Args:
        outdir: the QE scratch directory projwfc.x wrote into.
        prefix: the QE prefix.
        fermi_ev: Fermi level in eV, from the run that produced these files.
        element: which element's share to take. 'H' is the descriptor the Tc
            fit wants.

    Returns:
        H_DOS in [0, 1].

    Raises:
        FileNotFoundError: if projwfc.x wrote no per-orbital files.
        ValueError: if the total projected DOS at E_F is zero (an insulator, or
            a Fermi level outside the tabulated energy window).

    This is a projection onto atomic orbitals, so it inherits their implied
    radii - which the paper names as a likely source of the scatter in the
    H_DOS/Tc trend.
    """
    pattern = os.path.join(outdir, f"{prefix}.pdos_atm#*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No projwfc.x per-orbital files matching {pattern}. Did projwfc.x run?"
        )

    total = 0.0
    selected = 0.0
    for path in files:
        match = _PDOS_ATOM_FILE.search(path)
        if match is None:
            continue
        value = _ldos_at(path, fermi_ev)
        total += value
        if match.group(2) == element:
            selected += value

    if total <= 0.0:
        raise ValueError(
            f"Total projected DOS at E_F = {fermi_ev} eV is zero. Either the "
            f"compound is not metallic, or E_F lies outside the tabulated range."
        )
    return selected / total


def _ldos_at(path: str, energy_ev: float) -> float:
    """
    Linearly interpolate an ldos(E) column at one energy.

    projwfc.x writes '# E (eV)  ldos(E)  pdos(E) ...'; column 1 is the
    orbital-resolved local DOS, which is what sums to the atom's contribution.
    """
    table = np.loadtxt(path, comments="#")
    if table.ndim == 1:
        table = table.reshape(1, -1)
    energies, ldos = table[:, 0], table[:, 1]
    if energy_ev <= energies[0] or energy_ev >= energies[-1]:
        return 0.0
    return float(np.interp(energy_ev, energies, ldos))

