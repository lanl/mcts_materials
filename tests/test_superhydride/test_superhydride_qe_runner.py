"""
Unit tests for the Quantum ESPRESSO runner.

No real QE: the "binaries" are shell stubs, which is enough to pin the two
behaviours that matter - the JOB DONE gate and the per-run working directory -
plus the rank clamp that keeps small cells from aborting.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import os
import stat

import pytest

from mcts_framework.superhydride.qe.runner import JOB_DONE, QEError, QERunner


def make_stub(directory, name, body):
    """Write an executable shell stub that stands in for a QE binary."""
    path = directory / name
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def bin_dir(tmp_path):
    directory = tmp_path / "bin"
    directory.mkdir()
    return directory


@pytest.fixture
def runner(bin_dir):
    # No MPI launcher: the stubs are plain shell scripts.
    return QERunner(bin_dir=str(bin_dir), mpi_command="", ranks=1, timeout_s=60)


# --- The JOB DONE gate ----------------------------------------------------


def test_a_finished_step_returns_its_stdout(runner, bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", f'echo "all good"; echo "   {JOB_DONE}"')
    text = runner.run("pw.x", "&CONTROL\n/\n", str(tmp_path / "run"), stem="scf")
    assert "all good" in text
    assert JOB_DONE in text


def test_a_step_without_job_done_raises_even_on_exit_zero(runner, bin_dir, tmp_path):
    """
    pw.x exits 0 on several genuine failures - 'convergence NOT achieved' among
    them - so the exit code is not the signal.
    """
    make_stub(bin_dir, "pw.x", 'echo "convergence NOT achieved"; exit 0')
    with pytest.raises(QEError, match="did not print"):
        runner.run("pw.x", "&CONTROL\n/\n", str(tmp_path / "run"), stem="scf")


def test_the_failure_message_carries_the_tail_of_the_output(runner, bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", 'echo "Error in routine cdiaghg"; exit 1')
    with pytest.raises(QEError, match="cdiaghg"):
        runner.run("pw.x", "x", str(tmp_path / "run"), stem="scf")


def test_a_missing_binary_raises(runner, tmp_path):
    with pytest.raises(QEError, match="did not print"):
        runner.run("pw.x", "x", str(tmp_path / "run"), stem="scf")


def test_a_hanging_step_times_out(bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", "sleep 30")
    runner = QERunner(bin_dir=str(bin_dir), mpi_command="", ranks=1, timeout_s=1)
    with pytest.raises(QEError, match="exceeded"):
        runner.run("pw.x", "x", str(tmp_path / "run"), stem="scf")


# --- Files and directories ------------------------------------------------


def test_input_and_output_are_written_into_the_run_directory(runner, bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", f'echo "{JOB_DONE}"')
    workdir = tmp_path / "candidate"
    runner.run("pw.x", "&CONTROL\n  calculation = 'scf'\n/\n", str(workdir), stem="scf")
    assert (workdir / "scf.in").read_text().startswith("&CONTROL")
    assert JOB_DONE in (workdir / "scf.out").read_text()


def test_the_binary_runs_with_its_own_directory_as_cwd(runner, bin_dir, tmp_path):
    """
    pw.x resolves outdir against the process working directory. Two concurrent
    runs sharing one read each other's wavefunctions and return plausible
    numbers belonging to a different structure, with no error at all.
    """
    make_stub(bin_dir, "pw.x", f'pwd > where.txt; echo "{JOB_DONE}"')
    workdir = tmp_path / "candidate"
    runner.run("pw.x", "x", str(workdir), stem="scf")
    assert os.path.realpath((workdir / "where.txt").read_text().strip()) == os.path.realpath(
        workdir
    )


def test_two_candidates_do_not_share_a_directory(runner, bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", f'pwd > where.txt; echo "{JOB_DONE}"')
    runner.run("pw.x", "x", str(tmp_path / "a"), stem="scf")
    runner.run("pw.x", "x", str(tmp_path / "b"), stem="scf")
    assert (tmp_path / "a" / "where.txt").read_text() != (
        tmp_path / "b" / "where.txt"
    ).read_text()


def test_the_input_path_is_passed_to_the_binary(runner, bin_dir, tmp_path):
    make_stub(bin_dir, "pw.x", f'echo "args: $@"; echo "{JOB_DONE}"')
    text = runner.run("pw.x", "x", str(tmp_path / "run"), stem="nscf")
    assert "-input" in text
    assert "nscf.in" in text


def test_pencil_decomposition_is_requested_for_pw_only(runner, bin_dir, tmp_path):
    """-pd covers part of the FFT-plane shortage; pp.x and projwfc.x reject it."""
    make_stub(bin_dir, "pw.x", f'echo "args: $@"; echo "{JOB_DONE}"')
    make_stub(bin_dir, "pp.x", f'echo "args: $@"; echo "{JOB_DONE}"')
    assert "-pd" in runner.run("pw.x", "x", str(tmp_path / "a"), stem="scf")
    assert "-pd" not in runner.run("pp.x", "x", str(tmp_path / "b"), stem="pp")


# --- Environment ----------------------------------------------------------


def test_environment_setup_is_sourced_before_the_binary(bin_dir, tmp_path):
    """On a cluster the toolchain usually lives behind modules."""
    make_stub(bin_dir, "pw.x", f'echo "flag=$MY_TEST_FLAG"; echo "{JOB_DONE}"')
    runner = QERunner(
        bin_dir=str(bin_dir),
        mpi_command="",
        ranks=1,
        environment_setup="export MY_TEST_FLAG=loaded",
        timeout_s=60,
    )
    assert "flag=loaded" in runner.run("pw.x", "x", str(tmp_path / "run"), stem="scf")


def test_openmp_threading_is_pinned_off_by_default(runner, bin_dir, tmp_path):
    """QE's own threading fights the MPI decomposition."""
    make_stub(bin_dir, "pw.x", f'echo "omp=$OMP_NUM_THREADS"; echo "{JOB_DONE}"')
    assert "omp=1" in runner.run("pw.x", "x", str(tmp_path / "run"), stem="scf")


def test_preflight_reports_which_binaries_are_reachable(runner, bin_dir):
    make_stub(bin_dir, "pw.x", "true")
    found = runner.check_available()
    assert found["pw.x"] is True
    assert found["pp.x"] is False
    assert found["projwfc.x"] is False


# --- Rank clamping --------------------------------------------------------


def test_ranks_are_clamped_to_the_cells_fft_planes(make_superhydride_structure):
    """
    QE distributes FFT planes across ranks; a small cell has fewer planes than
    a node has cores, and the ranks that get none abort the run with "there are
    processes with no planes".
    """
    pytest.importorskip("ase")
    runner = QERunner(ranks=96)
    small = make_superhydride_structure(a=3.0).atoms
    large = make_superhydride_structure(a=12.0).atoms
    assert runner.ranks_for(small, 90.0) < runner.ranks_for(large, 90.0)
    assert runner.ranks_for(small, 90.0) < 96


def test_rank_clamp_never_returns_zero(make_superhydride_structure):
    pytest.importorskip("ase")
    runner = QERunner(ranks=8)
    assert runner.ranks_for(make_superhydride_structure(a=0.5).atoms, 10.0) >= 1


def test_a_large_cell_keeps_the_requested_ranks(make_superhydride_structure):
    pytest.importorskip("ase")
    runner = QERunner(ranks=4)
    assert runner.ranks_for(make_superhydride_structure(a=20.0).atoms, 90.0) == 4
