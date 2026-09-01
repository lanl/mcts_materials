"""
Unit tests for Quantum ESPRESSO output parsing.

The pw.x fixtures below are excerpted from a real H3S Im-3m run on this
cluster, so the strings are the ones QE 7.3.1 actually prints.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import numpy as np
import pytest

from mcts_framework.superhydride.qe.outputs import (
    hydrogen_dos_fraction,
    parse_pw_output,
    relaxed_atoms,
)

SCF_OUTPUT = """
     Program PWSCF v.7.3.1 starts on  1Sep2026 at 13:29:32

     Dense  grid:    11123 G-vectors     FFT dimensions: (  27,  27,  27)

     total energy              =     -50.10876793 Ry
     total energy              =     -50.11298351 Ry

     the Fermi energy is    17.6947 ev

!    total energy              =     -50.11298474 Ry
     estimated scf accuracy    <          4.4E-11 Ry

          total   stress  (Ry/bohr**3)                   (kbar)     P=     1997.84

   JOB DONE.
"""

VC_RELAX_OUTPUT = """
     Program PWSCF v.7.3.1 starts on  1Sep2026

!    total energy              =     -50.05000000 Ry
          total   stress  (Ry/bohr**3)                   (kbar)     P=     2050.10

     End of BFGS Geometry Optimization

Begin final coordinates
     new unit-cell volume =    179.36085 a.u.^3 (    26.57856 Ang^3 )

CELL_PARAMETERS (angstrom)
   3.0100000000   0.0000000000   0.0000000000
   0.0000000000   3.0100000000   0.0000000000
   0.0000000000   0.0000000000   3.0100000000

ATOMIC_POSITIONS (crystal)
S             0.0000000000        0.0000000000        0.0000000000
S             0.5000000000        0.5000000000        0.5000000000
H             0.5000000000        0.0000000000        0.0000000000
End final coordinates

     A final scf calculation at the relaxed structure.

     the Fermi energy is    17.1000 ev

!    total energy              =     -50.12000000 Ry
          total   stress  (Ry/bohr**3)                   (kbar)     P=     1998.20

   JOB DONE.
"""


# --- The JOB DONE gate ----------------------------------------------------


def test_a_finished_run_is_usable():
    result = parse_pw_output(SCF_OUTPUT)
    assert result.job_done
    assert result.ok()
    assert result.failure_reason() == ""


def test_a_run_without_job_done_is_not_usable():
    """pw.x exits 0 on several genuine failures, so the footer is the gate."""
    result = parse_pw_output(SCF_OUTPUT.replace("JOB DONE.", ""))
    assert not result.job_done
    assert not result.ok()
    assert "JOB DONE" in result.failure_reason()


def test_unconverged_scf_is_rejected_even_though_it_says_job_done():
    text = SCF_OUTPUT.replace(
        "estimated scf accuracy", "convergence NOT achieved\n     estimated scf accuracy"
    )
    result = parse_pw_output(text)
    assert result.job_done
    assert not result.scf_converged
    assert not result.ok()
    assert "convergence" in result.failure_reason().lower()


# --- Values -------------------------------------------------------------


def test_scf_values():
    result = parse_pw_output(SCF_OUTPUT)
    assert result.energy_ry == pytest.approx(-50.11298474)
    assert result.pressure_kbar == pytest.approx(1997.84)
    assert result.pressure_gpa == pytest.approx(199.784)
    assert result.fermi_ev == pytest.approx(17.6947)


def test_a_relax_quotes_the_final_scf_not_the_one_bfgs_steered_by():
    """
    The values BFGS quotes carry a Pulay error - the basis no longer fits the
    cell it moved to. The last ones in the file are on the relaxed cell in a
    basis that does.
    """
    result = parse_pw_output(VC_RELAX_OUTPUT)
    assert result.pressure_kbar == pytest.approx(1998.20)   # not 2050.10
    assert result.energy_ry == pytest.approx(-50.12)        # not -50.05
    assert result.fermi_ev == pytest.approx(17.1000)


# --- Geometry -------------------------------------------------------------


def test_relaxed_geometry_is_parsed():
    result = parse_pw_output(VC_RELAX_OUTPUT)
    assert result.bfgs_converged is True
    assert result.ok(require_relaxed=True)
    assert result.final_cell == pytest.approx(np.diag([3.01, 3.01, 3.01]))
    assert result.final_symbols == ["S", "S", "H"]
    assert result.final_scaled_positions == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]])
    )


def test_a_single_point_run_reports_no_geometry_optimisation():
    result = parse_pw_output(SCF_OUTPUT)
    assert result.bfgs_converged is None
    assert result.final_cell is None
    assert result.ok()                          # fine as a single point
    assert not result.ok(require_relaxed=True)  # but not as a relaxation


def test_a_relax_that_exhausts_nstep_is_rejected():
    """
    It still prints JOB DONE. and writes no final-coordinates block at all.
    """
    text = VC_RELAX_OUTPUT.replace("End of BFGS Geometry Optimization", "")
    start = text.find("Begin final coordinates")
    end = text.find("End final coordinates") + len("End final coordinates")
    text = text[:start] + text[end:] + "\n     BFGS Geometry Optimization\n"
    result = parse_pw_output(text)
    assert result.job_done
    assert not result.ok(require_relaxed=True)
    assert "did not converge" in result.failure_reason(require_relaxed=True)


def test_bfgs_failure_is_rejected_despite_a_final_coordinates_block():
    text = VC_RELAX_OUTPUT.replace(
        "End of BFGS Geometry Optimization",
        "bfgs failed after 100 scf cycles ... convergence not achieved",
    )
    result = parse_pw_output(text)
    assert result.bfgs_converged is False
    assert not result.ok(require_relaxed=True)


def test_relaxed_atoms_applies_the_geometry_to_a_template(make_superhydride_structure):
    pytest.importorskip("ase")
    template = make_superhydride_structure().atoms[:3]
    result = parse_pw_output(VC_RELAX_OUTPUT)
    updated = relaxed_atoms(result, template)
    assert np.asarray(updated.get_cell()) == pytest.approx(np.diag([3.01, 3.01, 3.01]))
    assert updated.get_scaled_positions() == pytest.approx(result.final_scaled_positions)
    # The template is untouched.
    assert np.asarray(template.get_cell())[0, 0] == pytest.approx(5.0)


def test_relaxed_atoms_refuses_a_single_point_result(make_superhydride_structure):
    pytest.importorskip("ase")
    with pytest.raises(ValueError, match="no relaxed structure"):
        relaxed_atoms(parse_pw_output(SCF_OUTPUT), make_superhydride_structure().atoms)


# --- Projected DOS --------------------------------------------------------


def write_pdos(directory, prefix, atom_index, element, ldos_by_energy):
    """Write one projwfc.x per-orbital file: '# E (eV)  ldos(E)  pdos(E)'."""
    path = directory / f"{prefix}.pdos_atm#{atom_index}({element})_wfc#1(s)"
    lines = ["# E (eV)  ldos(E)  pdos(E)"]
    for energy, ldos in ldos_by_energy:
        lines.append(f"  {energy:.3f}  {ldos:.6f}  {ldos:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_hydrogen_dos_fraction_is_a_share_of_the_projections(tmp_path):
    grid = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
    write_pdos(tmp_path, "sh", 1, "H", [(e, 3.0 * v) for e, v in grid])
    write_pdos(tmp_path, "sh", 2, "H", [(e, 1.0 * v) for e, v in grid])
    write_pdos(tmp_path, "sh", 3, "S", [(e, 4.0 * v) for e, v in grid])
    # H contributes 3 + 1 of the total 8 at E_F = 0.
    assert hydrogen_dos_fraction(str(tmp_path), "sh", 0.0) == pytest.approx(0.5)


def test_hydrogen_dos_fraction_interpolates_between_grid_points(tmp_path):
    write_pdos(tmp_path, "sh", 1, "H", [(0.0, 0.0), (1.0, 2.0)])
    write_pdos(tmp_path, "sh", 2, "S", [(0.0, 2.0), (1.0, 2.0)])
    # At E = 0.5: H = 1.0, S = 2.0.
    assert hydrogen_dos_fraction(str(tmp_path), "sh", 0.5) == pytest.approx(1.0 / 3.0)


def test_hydrogen_dos_fraction_is_bounded(tmp_path):
    write_pdos(tmp_path, "sh", 1, "H", [(-1.0, 0.0), (0.0, 5.0), (1.0, 0.0)])
    assert hydrogen_dos_fraction(str(tmp_path), "sh", 0.0) == pytest.approx(1.0)


def test_missing_projwfc_files_are_an_error_not_a_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="No projwfc.x per-orbital files"):
        hydrogen_dos_fraction(str(tmp_path), "sh", 0.0)


def test_a_fermi_level_outside_the_tabulated_window_is_an_error(tmp_path):
    """
    Silently returning 0.0 would look like an insulator and score the candidate
    as if it had been computed.
    """
    write_pdos(tmp_path, "sh", 1, "H", [(-1.0, 1.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match="outside the tabulated range"):
        hydrogen_dos_fraction(str(tmp_path), "sh", 99.0)
