"""
ELF-based descriptors: the networking value phi and the molecularity index phi*.

Both ask one question - given the region {ELF >= t}, what is connected to
what? - at different thresholds:

    phi   the largest t at which one connected component spans the crystal in
          all three directions (Belli et al., Nat. Commun. 12, 5381 (2021)).
    phi*  the largest t at which two hydrogen atoms fall in one component
          (Di Mauro et al.; used in Eq. 2 of Ann. Phys. 2025, 537, e00280).

phi* -> 1 is the signature of intact H2 molecules, and the molecularity factor
(phi*^2 - phi*^3) in the Tc fit vanishes there by construction.

Both are percolation thresholds evaluated on the SCF's FFT grid, so the grid
resolution is a floor on their precision - and it moves with the plane-wave
cutoff, which means descriptors computed at different cutoffs are not
comparable.

scipy is imported lazily so the rest of the package (the Tc fit, the reward)
stays importable without it.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np

from .elements import HYDROGEN

#: Default bisection tolerance on the ELF threshold. Finer than this is not
#: meaningful against the grid resolution of a typical screening SCF.
DEFAULT_TOL = 2e-3


@dataclass(frozen=True)
class ELFDescriptors:
    """The four inputs the Belli Tc fit needs, minus H_DOS (which comes from a PDOS)."""

    phi: float                              # networking value
    phi_star: float                         # molecularity index
    h_f: float                              # hydrogen fraction N_H / N_total
    grid_shape: Tuple[int, int, int]
    threshold_resolution: float


def hydrogen_fraction(atomic_numbers: Iterable[int]) -> float:
    """
    Return H_f = N_H / N_total from a list of atomic numbers.

    Raises:
        ValueError: on an empty structure.
    """
    numbers = [int(z) for z in atomic_numbers]
    if not numbers:
        raise ValueError("Cannot compute a hydrogen fraction for an empty structure")
    return sum(1 for z in numbers if z == HYDROGEN) / len(numbers)


# --- The two percolation thresholds ---------------------------------------


def networking_value(
    elf: np.ndarray,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    tol: float = DEFAULT_TOL,
) -> float:
    """
    Highest ELF isovalue whose isosurface spans the cell in all three directions.

    Args:
        elf: ELF sampled on a regular periodic grid, shape ``(nx, ny, nz)``.
        lo, hi: bracketing thresholds; ``lo`` must percolate and ``hi`` must not.
        tol: bisection tolerance on the ELF value.

    Returns:
        The networking value phi, or ``lo`` if nothing percolates above it.

    Monotonicity is what makes bisection valid: ``{ELF >= t}`` shrinks as ``t``
    grows, so percolation can only switch off once as the threshold rises.
    """
    return _bisect_threshold(
        lambda t: _spans_all_directions(elf >= t), lo=lo, hi=hi, tol=tol
    )


def molecularity_index(
    elf: np.ndarray,
    h_voxels: np.ndarray,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    tol: float = DEFAULT_TOL,
) -> float:
    """
    Highest ELF isovalue at which two hydrogen atoms become connected.

    Args:
        elf: ELF on a regular periodic grid.
        h_voxels: integer array ``(n_H, 3)`` of grid indices of the H nuclei.
        lo, hi, tol: as in :func:`networking_value`.

    Returns:
        The molecularity index phi*. Zero only if the cell has no hydrogen.

    Connectivity is evaluated across periodic images, so a *single* H in the
    primitive cell still has a finite phi* by bonding to its own image - in the
    crystal those are two distinct atoms. Requiring two H atoms *in the cell*
    would report phi* = 0 for high-symmetry single-H structures, and the
    (phi*^2 - phi*^3) factor would then silently collapse their predicted Tc to
    the intercept.
    """
    h_voxels = np.asarray(h_voxels, dtype=int)
    if h_voxels.shape[0] < 1:
        return 0.0
    return _bisect_threshold(
        lambda t: _any_pair_connected(elf >= t, h_voxels), lo=lo, hi=hi, tol=tol
    )


def compute_descriptors(
    elf: np.ndarray,
    h_frac_coords: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    tol: float = DEFAULT_TOL,
) -> ELFDescriptors:
    """
    Compute phi, phi* and H_f together.

    Args:
        elf: ELF on a regular periodic grid, shape ``(nx, ny, nz)``.
        h_frac_coords: ``(n_H, 3)`` fractional coordinates of the hydrogen
            atoms. Taken from the structure, not from the cube - see
            :func:`read_elf_cube` for why the cube's own atom block is unsafe.
        atomic_numbers: atomic numbers of every atom in the cell, for H_f.
        tol: bisection tolerance on the ELF threshold.
    """
    h_voxels = fractional_to_voxel(h_frac_coords, elf.shape)
    return ELFDescriptors(
        phi=networking_value(elf, tol=tol),
        phi_star=molecularity_index(elf, h_voxels, tol=tol),
        h_f=hydrogen_fraction(atomic_numbers),
        grid_shape=(elf.shape[0], elf.shape[1], elf.shape[2]),
        threshold_resolution=tol,
    )


# --- Internals ------------------------------------------------------------


def _bisect_threshold(
    predicate: Callable[[float], bool], *, lo: float, hi: float, tol: float
) -> float:
    """Largest t in [lo, hi] with predicate(t) true, for a monotone-decreasing predicate."""
    if not predicate(lo):
        # Not even the loosest threshold connects: there is nothing to find.
        return lo
    if predicate(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


class _PeriodicComponents:
    """
    Connected components of a periodic mask, with the period vectors of each.

    Tiling the mask and running a plain connected-component pass is not correct
    in general: a wrapping path may wander further than the tiling is wide, and
    how far it wanders depends on the cell you happened to choose - which makes
    phi depend on the box drawn around the crystal rather than on the crystal.

    So this does it exactly, in two steps:

    1. Label the mask *within* one cell (non-periodic) - a fast C-level pass
       that usually yields a handful of components.
    2. Union those components across the periodic faces, carrying a
       displacement vector per merge. When an edge closes a loop, the mismatch
       in accumulated displacement is a **period vector** of that component.

    A component spans the crystal in three dimensions exactly when its period
    vectors have rank 3. That statement is lattice-invariant, and it also
    catches what tiling is blind to: a network percolating only within a plane
    has rank-2 periods and is not three-dimensional.

    Connectivity is face-only (6-connectivity). 26-connectivity would let
    corner contacts count as bonding paths and percolate spuriously.
    """

    def __init__(self, mask: np.ndarray):
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(3, 1)  # face-sharing only
        self.labels, self.n = ndimage.label(mask, structure=structure)

        # Union-find over in-cell component labels, each carrying an integer
        # offset (in cells) relative to its parent.
        self._parent = np.arange(self.n + 1)
        self._offset = np.zeros((self.n + 1, 3), dtype=np.int64)
        self._periods: dict = {}

        for axis in (0, 1, 2):
            self._union_across_face(axis)

    # -- union-find with displacements ------------------------------------

    def _find(self, x: int) -> Tuple[int, np.ndarray]:
        """Root of `x` and the offset from `x` to that root, with path compression."""
        offset = np.zeros(3, dtype=np.int64)
        root = x
        while self._parent[root] != root:
            offset += self._offset[root]
            root = self._parent[root]

        node, acc = x, offset.copy()
        while self._parent[node] != node:
            nxt = self._parent[node]
            nxt_off = acc - self._offset[node]
            self._parent[node] = root
            self._offset[node] = acc
            node, acc = nxt, nxt_off
        return root, offset

    def _union(self, a: int, b: int, displacement: np.ndarray) -> None:
        """Record that component `a` touches component `b` shifted by `displacement` cells."""
        root_a, off_a = self._find(a)
        root_b, off_b = self._find(b)
        if root_a == root_b:
            # A loop. The residual displacement is a period vector.
            loop = off_a + displacement - off_b
            if loop.any():
                self._periods.setdefault(root_a, []).append(loop)
            return
        self._parent[root_b] = root_a
        self._offset[root_b] = off_a + displacement - off_b
        if root_b in self._periods:
            self._periods.setdefault(root_a, []).extend(self._periods.pop(root_b))

    def _union_across_face(self, axis: int) -> None:
        """Union components touching across the periodic boundary along `axis`."""
        last = np.take(self.labels, indices=-1, axis=axis)
        first = np.take(self.labels, indices=0, axis=axis)
        touching = (last > 0) & (first > 0)
        if not touching.any():
            return
        displacement = np.zeros(3, dtype=np.int64)
        displacement[axis] = 1
        pairs = np.unique(np.stack([last[touching], first[touching]], axis=1), axis=0)
        for a, b in pairs:
            self._union(int(a), int(b), displacement)

    # -- queries -----------------------------------------------------------

    def root_of(self, label: int) -> int:
        return self._find(int(label))[0] if label > 0 else 0

    def spans_3d(self) -> bool:
        """True if any component's period vectors span three dimensions."""
        for vectors in self._periods.values():
            if len(vectors) >= 3 and np.linalg.matrix_rank(np.array(vectors)) == 3:
                return True
        return False

    def wraps(self, label: int) -> bool:
        """True if this component connects to any of its own periodic images."""
        return bool(self._periods.get(self.root_of(label)))


def _spans_all_directions(mask: np.ndarray) -> bool:
    """True if a connected component spans the crystal in all three dimensions."""
    if not mask.any():
        return False
    return _PeriodicComponents(mask).spans_3d()


def _any_pair_connected(mask: np.ndarray, h_voxels: np.ndarray) -> bool:
    """True if two H sites share a component, counting periodic images as distinct atoms."""
    if not mask.any():
        return False

    components = _PeriodicComponents(mask)
    nx, ny, nz = mask.shape
    idx = (h_voxels[:, 0] % nx, h_voxels[:, 1] % ny, h_voxels[:, 2] % nz)
    site_labels = components.labels[idx]

    roots = [components.root_of(int(lab)) for lab in site_labels if lab > 0]
    if not roots:
        return False

    # Two distinct H atoms in one component.
    if len(roots) >= 2 and len(set(roots)) < len(roots):
        return True

    # An H bonded to its own periodic image is a real H-H contact in the crystal.
    return any(components.wraps(int(lab)) for lab in site_labels if lab > 0)


# --- Grid I/O -------------------------------------------------------------


def fractional_to_voxel(frac: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Map fractional coordinates onto grid indices, wrapping into the cell."""
    frac = np.asarray(frac, dtype=float) % 1.0
    dims = np.array(shape[:3])
    return np.rint(frac * dims).astype(int) % dims


def read_elf_cube(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a Gaussian cube as written by QE ``pp.x`` with ``output_format = 6``.

    Returns:
        ``(data, origin, cell, atoms)`` where ``data`` has shape
        ``(nx, ny, nz)``, ``cell`` is the 3x3 matrix of grid *span* vectors in
        bohr, and ``atoms`` holds one row per atom: the value in the cube's
        atomic-number column followed by Cartesian coordinates in bohr.

    Note on that first column: a Gaussian cube reserves it for the atomic
    number, and Quantum ESPRESSO writes the **species index** there instead -
    in an H3S cube written by pp.x, sulphur is 0 and hydrogen is 1. Never infer
    which atoms are hydrogen from it. Take the hydrogen coordinates from the
    structure you submitted; :func:`compute_descriptors` requires them.
    """
    with open(path) as fh:
        fh.readline()  # two comment lines
        fh.readline()

        tokens = fh.readline().split()
        natoms, origin = int(tokens[0]), np.array([float(x) for x in tokens[1:4]])

        shape, vectors = [], []
        for _ in range(3):
            tokens = fh.readline().split()
            shape.append(int(tokens[0]))
            vectors.append([float(x) for x in tokens[1:4]])

        atoms = []
        for _ in range(abs(natoms)):
            tokens = fh.readline().split()
            atoms.append([float(tokens[0]), *(float(x) for x in tokens[2:5])])

        values = np.fromstring(fh.read().replace("\n", " "), sep=" ")

    n_points = shape[0] * shape[1] * shape[2]
    if values.size < n_points:
        raise ValueError(
            f"{path}: expected {n_points} grid values for a "
            f"{shape[0]}x{shape[1]}x{shape[2]} grid, found {values.size}"
        )
    data = values[:n_points].reshape((shape[0], shape[1], shape[2]))
    # Row i of `vectors` is the per-voxel step; the cell edge is step * n_points.
    cell = np.array(vectors) * np.array(shape)[:, None]
    return data, origin, cell, np.array(atoms)


def cartesian_to_fractional(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to fractional ones for a row-vector `cell`."""
    return np.linalg.solve(np.asarray(cell).T, np.asarray(positions).T).T % 1.0


def descriptors_from_cube(
    cube_path: str,
    h_frac_coords: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    tol: float = DEFAULT_TOL,
) -> ELFDescriptors:
    """
    Read an ELF cube and compute phi, phi* and H_f.

    Args:
        cube_path: Gaussian cube of the ELF from ``pp.x`` (``plot_num = 8``).
        h_frac_coords: fractional coordinates of the hydrogen atoms, from the
            structure that was submitted.
        atomic_numbers: atomic numbers of every atom in the cell.
        tol: bisection tolerance on the ELF threshold.
    """
    elf, _origin, _cell, _atoms = read_elf_cube(cube_path)
    return compute_descriptors(elf, h_frac_coords, atomic_numbers, tol=tol)
