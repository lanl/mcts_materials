"""
Running Quantum ESPRESSO binaries.

Thin, deliberately: it writes an input, runs one executable in one working
directory, and hands back the captured stdout. The two things it refuses to
leave to chance are the ones that silently corrupt a campaign:

    * **``JOB DONE.`` gates every step.** pw.x exits 0 on several genuine
      failures, so the exit code is not the signal.
    * **One run, one working directory and one outdir.** pw.x resolves
      ``outdir`` relative to the process working directory, so concurrent runs
      that do not cd somewhere private share ``./tmp/<prefix>.save`` and read
      each other's wavefunctions. Nothing errors; every number looks plausible
      and belongs to a different structure.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

JOB_DONE = "JOB DONE."


class QEError(RuntimeError):
    """A Quantum ESPRESSO step that did not finish cleanly."""


@dataclass
class QERunner:
    """
    Invokes pw.x / pp.x / projwfc.x.

    Args:
        bin_dir: directory holding the QE executables. Empty means "on PATH".
        mpi_command: launcher, e.g. 'mpirun'. Empty runs the binary serially.
        ranks: MPI rank count. **This must follow the cell**: QE distributes FFT
            planes across ranks, and a small primitive cell has fewer planes
            than a modern node has cores - the ranks that get none abort the run
            with "there are processes with no planes". A campaign spanning
            4-atom and 44-atom cells cannot use one fixed number; see
            :meth:`ranks_for`.
        environment_setup: shell snippet sourced before the binary, for clusters
            where the toolchain lives behind modules. On Artemis, QE 7.3.1 is
            built against gcc 13.2.0 and OpenMPI 4.1.6:
            'module load gcc/13.2.0 openmpi/4.1.6'.
        extra_env: environment variables to set. OMP_NUM_THREADS=1 is set by
            default - QE's own threading fights the MPI decomposition.
        timeout_s: wall-clock limit per step.
    """

    bin_dir: str = ""
    mpi_command: str = "mpirun"
    ranks: int = 4
    environment_setup: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=lambda: {"OMP_NUM_THREADS": "1"})
    timeout_s: float = 7200.0
    #: Passed to pw.x. Pencil decomposition covers part of the plane shortage.
    pencil_decomposition: bool = True

    def executable(self, binary: str) -> str:
        return os.path.join(self.bin_dir, binary) if self.bin_dir else binary

    def ranks_for(self, atoms: "object", ecutwfc: float) -> int:
        """
        Rank count capped by the number of FFT planes the cell actually has.

        The smooth-grid plane count along the third axis is roughly
        ``sqrt(ecutwfc) * |a3| / pi`` in atomic units. Asking for more ranks
        than that aborts the run, so this clamps to it.
        """
        import numpy as np

        bohr_per_angstrom = 1.8897261254535
        a3 = float(np.linalg.norm(np.asarray(atoms.get_cell())[2])) * bohr_per_angstrom
        planes = max(int((ecutwfc**0.5) * a3 / 3.141592653589793), 1)
        return max(min(self.ranks, planes), 1)

    def run(
        self,
        binary: str,
        input_text: str,
        workdir: str,
        *,
        stem: str,
        ranks: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        """
        Write ``<stem>.in``, run ``binary``, and return its stdout.

        Args:
            binary: 'pw.x', 'pp.x' or 'projwfc.x'.
            input_text: the input deck.
            workdir: the run's private working directory (created if absent).
            stem: basename for the .in/.out pair.
            ranks: override the rank count for this step.
            extra_args: extra command-line flags for the binary.

        Returns:
            The captured stdout, which is also written to ``<stem>.out``.

        Raises:
            QEError: if the binary is missing, times out, or the output does not
                end in 'JOB DONE.'.
        """
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / f"{stem}.in"
        output_path = directory / f"{stem}.out"
        input_path.write_text(input_text)

        argv = []
        if self.mpi_command:
            argv += shlex.split(self.mpi_command) + ["-np", str(ranks or self.ranks)]
        argv.append(self.executable(binary))
        if binary == "pw.x" and self.pencil_decomposition:
            argv += ["-pd", ".true."]
        argv += extra_args or []
        argv += ["-input", str(input_path)]

        command = shlex.join(argv)
        if self.environment_setup:
            command = f"{self.environment_setup}\n{command}"

        environment = {**os.environ, **self.extra_env}
        logger.info("QE %s in %s (%s ranks)", binary, workdir, ranks or self.ranks)

        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(directory),
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise QEError(
                f"{binary} in {workdir} exceeded {self.timeout_s} s"
            ) from exc
        except FileNotFoundError as exc:  # pragma: no cover - missing bash
            raise QEError(f"Could not launch {binary}: {exc}") from exc

        text = completed.stdout or ""
        output_path.write_text(text + (completed.stderr or ""))

        if JOB_DONE not in text:
            tail = "\n".join((text + (completed.stderr or "")).splitlines()[-15:])
            raise QEError(
                f"{binary} in {workdir} did not print '{JOB_DONE}' "
                f"(exit code {completed.returncode}). Last lines:\n{tail}"
            )
        return text

    def check_available(self, binaries=("pw.x", "pp.x", "projwfc.x")) -> Dict[str, bool]:
        """
        Report which executables this runner can actually find and start.

        Useful as a preflight before a long campaign, and as the thing to check
        when a search returns nothing but zero rewards.
        """
        found: Dict[str, bool] = {}
        for binary in binaries:
            command = f"command -v {shlex.quote(self.executable(binary))}"
            if self.environment_setup:
                command = f"{self.environment_setup}\n{command}"
            try:
                completed = subprocess.run(
                    ["bash", "-lc", command], capture_output=True, text=True, timeout=120
                )
                found[binary] = completed.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):  # pragma: no cover
                found[binary] = False
        return found
