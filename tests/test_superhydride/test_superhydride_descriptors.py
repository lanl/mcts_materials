"""
Unit tests for the ELF descriptors phi and phi*.

Both are percolation thresholds, so they are tested on synthetic ELF fields
whose connectivity is known by construction: a network of rods at a high ELF
value on a low background, where the answer is "the rod value" when the rods
percolate and "the background value" when they do not.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from pathlib import Path

import numpy as np
import pytest

from mcts_framework.superhydride import descriptors

pytest.importorskip("scipy", reason="ELF descriptors need scipy.ndimage")

GRID = 12
BACKGROUND = 0.10
ROD = 0.90
TOL = 2e-3


def _field(*, axes):
    """
    An ELF field with rods of value ROD along the given axes through the origin.

    Rods along all three axes give a component whose period vectors span three
    dimensions; fewer axes give a lower-rank network that must NOT count as
    percolating.
    """
    elf = np.full((GRID, GRID, GRID), BACKGROUND)
    for axis in axes:
        index = [0, 0, 0]
        index[axis] = slice(None)
        elf[tuple(index)] = ROD
    return elf


def _approx_rod(value):
    """Bisection converges on the switch point from below, within one tolerance."""
    return ROD - TOL <= value <= ROD


def _approx_background(value):
    return BACKGROUND - TOL <= value <= BACKGROUND


# --- Networking value phi -------------------------------------------------


def test_phi_finds_a_network_spanning_all_three_directions():
    assert _approx_rod(descriptors.networking_value(_field(axes=(0, 1, 2)), tol=TOL))


@pytest.mark.parametrize("axes", [(0,), (0, 1), (1, 2)])
def test_phi_rejects_networks_of_lower_dimension(axes):
    """
    A network percolating along one axis or within a plane is not a
    three-dimensional network. Its period vectors have rank 1 or 2, so phi
    falls back to the background - the highest threshold at which the whole
    cell is still one connected block.
    """
    assert _approx_background(descriptors.networking_value(_field(axes=axes), tol=TOL))


def test_phi_is_zero_when_nothing_connects():
    """A field below the bracket everywhere percolates nowhere."""
    elf = np.zeros((GRID, GRID, GRID))
    assert descriptors.networking_value(elf, lo=0.5, hi=1.0, tol=TOL) == 0.5


def test_phi_does_not_depend_on_the_cell_chosen():
    """
    phi is a property of the crystal, so tiling the cell must not change it.
    (Tiling the mask and running a plain component pass does change it, which
    is why the implementation uses period vectors instead.)
    """
    elf = _field(axes=(0, 1, 2))
    tiled = np.tile(elf, (2, 2, 2))
    assert descriptors.networking_value(tiled, tol=TOL) == pytest.approx(
        descriptors.networking_value(elf, tol=TOL), abs=2 * TOL
    )


def _two_site_field(second):
    """Background everywhere, ROD at (3,3,3) and at `second`."""
    elf = np.full((GRID, GRID, GRID), BACKGROUND)
    elf[3, 3, 3] = ROD
    elf[second] = ROD
    return elf, np.array([[3, 3, 3], list(second)])


@pytest.mark.parametrize(
    "second,label",
    [((4, 4, 3), "edge diagonal"), ((4, 4, 4), "corner diagonal")],
)
def test_diagonal_contacts_are_not_bonds(second, label):
    """
    Connectivity must be face-only. 26-connectivity would let these corner and
    edge contacts count as bonding paths and percolate spuriously, so the two
    sites here must stay in separate components until the background joins them.
    """
    elf, h_voxels = _two_site_field(second)
    assert _approx_background(descriptors.molecularity_index(elf, h_voxels, tol=TOL)), label


def test_face_contacts_are_bonds():
    """The control for the test above: one step along an axis does connect."""
    elf, h_voxels = _two_site_field((4, 3, 3))
    assert _approx_rod(descriptors.molecularity_index(elf, h_voxels, tol=TOL))


# --- Molecularity index phi* ----------------------------------------------


def test_phi_star_connects_two_hydrogens_in_one_rod():
    elf = _field(axes=(0,))
    h_voxels = np.array([[2, 0, 0], [5, 0, 0]])  # both inside the rod
    assert _approx_rod(descriptors.molecularity_index(elf, h_voxels, tol=TOL))


def test_phi_star_falls_back_when_hydrogens_are_in_separate_components():
    elf = np.full((GRID, GRID, GRID), BACKGROUND)
    elf[2, 2, 2] = ROD
    elf[8, 8, 8] = ROD
    h_voxels = np.array([[2, 2, 2], [8, 8, 8]])
    # They only share a component once the background is included.
    assert _approx_background(descriptors.molecularity_index(elf, h_voxels, tol=TOL))


def test_single_hydrogen_bonds_to_its_own_periodic_image():
    """
    One H in the primitive cell still has a finite phi*: in the crystal it and
    its image are two distinct atoms. Reporting 0 here would collapse a
    high-symmetry structure's predicted Tc to the intercept.
    """
    elf = _field(axes=(0,))
    h_voxels = np.array([[3, 0, 0]])
    assert _approx_rod(descriptors.molecularity_index(elf, h_voxels, tol=TOL))


def test_isolated_hydrogen_does_not_wrap():
    """An H in a blob that touches no image connects only via the background."""
    elf = np.full((GRID, GRID, GRID), BACKGROUND)
    elf[5, 5, 5] = ROD
    assert _approx_background(descriptors.molecularity_index(elf, np.array([[5, 5, 5]]), tol=TOL))


def test_phi_star_is_zero_without_hydrogen():
    assert descriptors.molecularity_index(_field(axes=(0, 1, 2)), np.empty((0, 3)), tol=TOL) == 0.0


def test_phi_equals_phi_star_for_a_uniform_network():
    """
    Expected, not suspicious: when every H-H contact is equivalent the whole
    network opens at one threshold. The two diverge only when the hydrogen
    interactions are inhomogeneous, which is what phi* exists to detect.
    """
    elf = _field(axes=(0, 1, 2))
    h_voxels = np.array([[3, 0, 0], [0, 3, 0]])
    assert descriptors.networking_value(elf, tol=TOL) == pytest.approx(
        descriptors.molecularity_index(elf, h_voxels, tol=TOL), abs=1e-9
    )


def test_inhomogeneous_interactions_make_phi_star_exceed_phi():
    """
    A strong H-H contact (0.95) inside a weaker percolating network (0.6):
    phi* reports the strong contact, phi the network. This gap is the
    molecularity the Tc fit penalises.
    """
    elf = np.full((GRID, GRID, GRID), BACKGROUND)
    for axis in (0, 1, 2):
        index = [0, 0, 0]
        index[axis] = slice(None)
        elf[tuple(index)] = 0.60
    elf[0, 0, 0] = 0.95
    elf[1, 0, 0] = 0.95
    h_voxels = np.array([[0, 0, 0], [1, 0, 0]])

    phi = descriptors.networking_value(elf, tol=TOL)
    phi_star = descriptors.molecularity_index(elf, h_voxels, tol=TOL)
    assert 0.60 - TOL <= phi <= 0.60
    assert 0.95 - TOL <= phi_star <= 0.95
    assert phi_star > phi


# --- Hydrogen fraction ----------------------------------------------------


def test_hydrogen_fraction():
    assert descriptors.hydrogen_fraction([57, 4] + [1] * 8) == pytest.approx(0.8)   # LaBeH8
    assert descriptors.hydrogen_fraction([20] + [1] * 6) == pytest.approx(6 / 7)    # CaH6
    assert descriptors.hydrogen_fraction([46, 29, 1, 1]) == pytest.approx(0.5)      # PdCuH2


def test_hydrogen_fraction_rejects_an_empty_structure():
    with pytest.raises(ValueError, match="empty structure"):
        descriptors.hydrogen_fraction([])


# --- Grid indexing --------------------------------------------------------


def test_fractional_to_voxel_wraps_into_the_cell():
    voxels = descriptors.fractional_to_voxel([[0.0, 0.5, 0.999], [-0.25, 1.25, 2.0]], (8, 8, 8))
    assert voxels.tolist() == [[0, 4, 0], [6, 2, 0]]


# --- Cube I/O -------------------------------------------------------------


def _write_cube(path, data, cell_bohr, atoms):
    """Minimal Gaussian cube writer, matching what pp.x emits."""
    nx, ny, nz = data.shape
    steps = np.array(cell_bohr) / np.array([nx, ny, nz])[:, None]
    lines = ["ELF", "test cube", f"{len(atoms):5d} 0.0 0.0 0.0"]
    for n, step in zip((nx, ny, nz), steps):
        lines.append(f"{n:5d} {step[0]:.6f} {step[1]:.6f} {step[2]:.6f}")
    for species_index, (x, y, z) in atoms:
        lines.append(f"{species_index:5d} 0.0 {x:.6f} {y:.6f} {z:.6f}")
    flat = data.reshape(-1)
    for start in range(0, flat.size, 6):
        lines.append(" ".join(f"{v:.6e}" for v in flat[start : start + 6]))
    Path(path).write_text("\n".join(lines) + "\n")


def test_cube_round_trip(tmp_path):
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6) / 120.0
    cell = np.diag([4.0, 5.0, 6.0])
    cube = tmp_path / "elf.cube"
    _write_cube(cube, data, cell, [(1, (0.0, 0.0, 0.0)), (0, (2.0, 2.5, 3.0))])

    read_data, _origin, read_cell, atoms = descriptors.read_elf_cube(str(cube))
    assert read_data.shape == (4, 5, 6)
    assert np.allclose(read_data, data)
    assert np.allclose(read_cell, cell)
    assert len(atoms) == 2


def test_cube_reader_rejects_a_truncated_grid(tmp_path):
    cube = tmp_path / "short.cube"
    cube.write_text(
        "ELF\ntest\n    1 0.0 0.0 0.0\n"
        "    4 1.0 0.0 0.0\n    4 0.0 1.0 0.0\n    4 0.0 0.0 1.0\n"
        "    1 0.0 0.0 0.0 0.0\n"
        "0.1 0.2 0.3\n"  # 3 values for a 64-point grid
    )
    with pytest.raises(ValueError, match="expected 64 grid values"):
        descriptors.read_elf_cube(str(cube))


def test_descriptors_from_cube_uses_supplied_hydrogen_positions(tmp_path):
    """
    Hydrogen comes from the structure, never from the cube's atom block: QE
    writes the species index in the column a cube reserves for Z.
    """
    elf = _field(axes=(0, 1, 2))
    cube = tmp_path / "elf.cube"
    # Declare the atoms with a deliberately wrong "atomic number" column.
    _write_cube(cube, elf, np.diag([12.0, 12.0, 12.0]), [(0, (0.0, 0.0, 0.0))])

    result = descriptors.descriptors_from_cube(
        str(cube),
        h_frac_coords=np.array([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        atomic_numbers=[57, 1, 1],
        tol=TOL,
    )
    assert _approx_rod(result.phi)
    assert _approx_rod(result.phi_star)
    assert result.h_f == pytest.approx(2 / 3)
    assert result.grid_shape == (GRID, GRID, GRID)
    assert result.threshold_resolution == TOL


def test_compute_descriptors_is_consistent_with_the_individual_functions():
    elf = _field(axes=(0, 1, 2))
    h_frac = np.array([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result = descriptors.compute_descriptors(elf, h_frac, [57, 4, 1, 1], tol=TOL)
    voxels = descriptors.fractional_to_voxel(h_frac, elf.shape)
    assert result.phi == descriptors.networking_value(elf, tol=TOL)
    assert result.phi_star == descriptors.molecularity_index(elf, voxels, tol=TOL)
    assert result.h_f == pytest.approx(0.5)
