"""
Unit tests for Quantum ESPRESSO input generation.

No QE binaries needed: these check the decks the funnel would submit, and the
two refusals that exist because both produce a file that runs and lies.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math

import numpy as np
import pytest

pytest.importorskip("ase", reason="QE input generation needs ASE")

from mcts_framework.superhydride.qe import inputs  # noqa: E402
from mcts_framework.superhydride.qe.inputs import (  # noqa: E402
    QESettings,
    hydrogen_fractional_coordinates,
    kgrid_from_spacing,
    write_pp_elf_input,
    write_projwfc_input,
    write_pw_input,
)


@pytest.fixture
def make_structure(make_superhydride_structure):
    return make_superhydride_structure


@pytest.fixture
def settings(tmp_path):
    return QESettings(pseudo_dir=str(tmp_path))


def namelist_value(deck, key):
    """Pull `key = value` out of a QE deck."""
    for line in deck.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key} ="):
            return stripped.split("=", 1)[1].strip()
    return None


# --- k-point meshes -------------------------------------------------------


def test_kgrid_follows_the_cell_not_a_fixed_integer():
    """
    A fixed n x n x n is coarse in a small cell and wasteful in a large one.
    Sampling at a fixed spacing gives a denser mesh for the smaller cell.
    """
    small = kgrid_from_spacing(np.diag([3.0, 3.0, 3.0]), 0.2262)
    large = kgrid_from_spacing(np.diag([9.0, 9.0, 9.0]), 0.2262)
    assert small > large
    assert small == (10, 10, 10)


def test_kgrid_distributes_by_reciprocal_vector_length():
    """An elongated cell gets fewer points along its long real-space axis."""
    n1, n2, n3 = kgrid_from_spacing(np.diag([3.0, 3.0, 6.0]), 0.2262)
    assert n1 == n2
    assert n3 < n1
    assert n3 == pytest.approx(n1 / 2, abs=1)


def test_kgrid_is_at_least_one_point():
    assert kgrid_from_spacing(np.diag([40.0, 40.0, 40.0]), 5.0) == (1, 1, 1)


def test_kgrid_rejects_a_non_positive_spacing():
    with pytest.raises(ValueError, match="must be positive"):
        kgrid_from_spacing(np.eye(3), 0.0)


def test_nscf_mesh_is_denser_than_the_scf_one(settings, make_structure):
    scf = write_pw_input(
        make_structure().atoms, settings, calculation="scf", prefix="p", outdir="./s"
    )
    nscf = write_pw_input(
        make_structure().atoms, settings, calculation="nscf", prefix="p", outdir="./s"
    )
    scf_grid = [int(x) for x in scf.split("K_POINTS automatic")[1].split()[:3]]
    nscf_grid = [int(x) for x in nscf.split("K_POINTS automatic")[1].split()[:3]]
    assert all(n >= s for n, s in zip(nscf_grid, scf_grid))
    assert nscf_grid != scf_grid


# --- pw.x decks -----------------------------------------------------------


def test_scf_deck_carries_the_protocol(settings, make_structure):
    deck = write_pw_input(
        make_structure().atoms, settings, calculation="scf", prefix="sh", outdir="./scratch"
    )
    assert namelist_value(deck, "calculation") == "'scf'"
    assert namelist_value(deck, "prefix") == "'sh'"
    assert namelist_value(deck, "outdir") == "'./scratch'"
    assert float(namelist_value(deck, "ecutwfc")) == settings.ecutwfc
    assert float(namelist_value(deck, "ecutrho")) == settings.ecutrho
    assert namelist_value(deck, "occupations") == "'smearing'"
    assert namelist_value(deck, "smearing") == "'mp'"
    assert float(namelist_value(deck, "degauss")) == 0.02


def test_a_hydride_under_pressure_is_treated_as_a_metal(settings, make_structure):
    """
    Fixed occupations fill a set number of bands and need a gap; below a 0.1%
    charge error pw.x converges to a wrong ground state and says nothing.
    """
    deck = write_pw_input(
        make_structure().atoms, settings, calculation="scf", prefix="p", outdir="./s"
    )
    assert "occupations = 'fixed'" not in deck
    assert namelist_value(deck, "occupations") == "'smearing'"


def test_deck_lists_every_species_and_atom(settings, make_structure):
    deck = write_pw_input(
        make_structure().atoms, settings, calculation="scf", prefix="p", outdir="./s"
    )
    assert namelist_value(deck, "nat") == "10"
    assert namelist_value(deck, "ntyp") == "3"
    species_block = deck.split("ATOMIC_SPECIES")[1].split("ATOMIC_POSITIONS")[0]
    assert "Be.upf" in species_block
    assert "La.upf" in species_block
    assert "H.upf" in species_block
    positions = deck.split("ATOMIC_POSITIONS crystal")[1].split("CELL_PARAMETERS")[0]
    assert len(positions.strip().splitlines()) == 10
    assert positions.count("  H  ") == 8


def test_explicit_pseudo_filenames_override_the_default_pattern(tmp_path, make_structure):
    settings = QESettings(
        pseudo_dir=str(tmp_path),
        pseudo_files={"H": "H_ONCV_PBE-1.0.oncvpsp.upf", "La": "La.paw.z_11.upf"},
    )
    deck = write_pw_input(
        make_structure().atoms, settings, calculation="scf", prefix="p", outdir="./s"
    )
    assert "H_ONCV_PBE-1.0.oncvpsp.upf" in deck
    assert "La.paw.z_11.upf" in deck
    assert "Be.upf" in deck  # unlisted element falls back to the pattern


def test_vc_relax_requires_a_target_pressure(settings, make_structure):
    """
    A variable-cell relaxation with no target relaxes to 0 GPa, which for a
    pressure-stabilised hydride is a different material.
    """
    with pytest.raises(ValueError, match="target pressure"):
        write_pw_input(
            make_structure().atoms,
            settings,
            calculation="vc-relax",
            prefix="p",
            outdir="./s",
        )


def test_vc_relax_writes_the_pressure_in_kbar(settings, make_structure):
    deck = write_pw_input(
        make_structure().atoms,
        settings,
        calculation="vc-relax",
        prefix="p",
        outdir="./s",
        pressure_gpa=200.0,
    )
    assert "&CELL" in deck and "&IONS" in deck
    assert float(namelist_value(deck, "press")) == 2000.0  # QE wants kbar


def test_single_point_decks_have_no_ions_or_cell_namelist(settings, make_structure):
    for calculation in ("scf", "nscf"):
        deck = write_pw_input(
            make_structure().atoms,
            settings,
            calculation=calculation,
            prefix="p",
            outdir="./s",
        )
        assert "&IONS" not in deck
        assert "&CELL" not in deck


def test_pseudo_dir_is_required(make_structure):
    with pytest.raises(ValueError, match="pseudopotential directory"):
        write_pw_input(
            make_structure().atoms,
            QESettings(pseudo_dir=""),
            calculation="scf",
            prefix="p",
            outdir="./s",
        )


def test_unknown_calculation_is_rejected(settings, make_structure):
    with pytest.raises(ValueError, match="Unsupported pw.x calculation"):
        write_pw_input(
            make_structure().atoms, settings, calculation="md", prefix="p", outdir="./s"
        )


def test_explicit_kgrid_overrides_the_spacing(settings, make_structure):
    deck = write_pw_input(
        make_structure().atoms,
        settings,
        calculation="scf",
        prefix="p",
        outdir="./s",
        kgrid=(3, 5, 7),
    )
    assert "3 5 7 0 0 0" in deck


# --- pp.x and projwfc.x ---------------------------------------------------


def test_pp_deck_asks_for_the_elf_as_a_cube():
    deck = write_pp_elf_input(
        prefix="sh", outdir="./scratch", filplot="elf.dat", fileout="elf.cube"
    )
    assert namelist_value(deck, "plot_num") == str(inputs.PP_PLOT_NUM_ELF) == "8"
    assert namelist_value(deck, "output_format") == str(inputs.PP_OUTPUT_FORMAT_CUBE)
    assert namelist_value(deck, "iflag") == "3"
    assert namelist_value(deck, "fileout") == "'elf.cube'"


def test_projwfc_deck_sets_filpdos_explicitly():
    """
    projwfc.x resolves filpdos against its working directory, not outdir, so
    the reader has to be told where the files land.
    """
    deck = write_projwfc_input(prefix="sh", outdir="./scratch", degauss=0.02)
    assert namelist_value(deck, "filpdos") == "'sh'"
    assert namelist_value(deck, "prefix") == "'sh'"
    assert float(namelist_value(deck, "degauss")) == 0.02


# --- structure helpers ----------------------------------------------------


def test_hydrogen_coordinates_come_from_the_structure(make_structure):
    """
    Never from the cube's atom block: QE writes the species index in the column
    a Gaussian cube reserves for the atomic number.
    """
    coords = hydrogen_fractional_coordinates(make_structure().atoms)
    assert coords.shape == (8, 3)
    assert np.all((coords >= 0.0) & (coords <= 1.0))
    assert np.allclose(np.unique(coords), [0.25, 0.75])


def test_ecut_ratio_reports_what_to_check_against_the_upf_type():
    assert QESettings().ecut_ratio() == 4.0
    assert QESettings(ecutwfc=90.0, ecutrho=900.0).ecut_ratio() == 10.0


def test_default_protocol_matches_the_papers_physical_choices():
    settings = QESettings()
    assert settings.ecutwfc == 90.0
    assert settings.degauss == 0.02
    assert settings.smearing == "mp"
    assert settings.kspacing_scf == pytest.approx(2 * math.pi * 0.036)
    assert settings.kspacing_nscf == pytest.approx(2 * math.pi * 0.018)
