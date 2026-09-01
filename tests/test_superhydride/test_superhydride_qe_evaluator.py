"""
Unit tests for the Quantum ESPRESSO evaluator and the funnel's guard rails.

The funnel itself is stubbed - what is under test is the contract around it:
one failed candidate costs one candidate, the cache resumes an interrupted
campaign and replays a finished one, and the two "different material" refusals
fire before any compute is spent.

An opt-in integration test at the end runs the real binaries; it is skipped
unless MCTS_QE_INTEGRATION is set.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math
import os

import pandas as pd
import pytest

pytest.importorskip("ase", reason="the QE evaluator needs ASE")
pytest.importorskip("spglib", reason="structure identifiers need spglib")

from mcts_framework.superhydride import DescriptorTableEvaluator, TcReward  # noqa: E402
from mcts_framework.superhydride.qe import (  # noqa: E402
    QEError,
    QERunner,
    QESettings,
    QuantumEspressoEvaluator,
    run_ground_state,
)
from mcts_framework.superhydride.qe import pipeline as pipeline_module  # noqa: E402
from mcts_framework.superhydride.qe.pipeline import GroundStateResult  # noqa: E402


@pytest.fixture
def make_structure(make_superhydride_structure):
    return make_superhydride_structure


@pytest.fixture
def settings(tmp_path):
    return QESettings(pseudo_dir=str(tmp_path))


def fake_result(phi=0.60, phi_star=0.70, h_f=0.8, h_dos=0.55):
    return GroundStateResult(
        phi=phi,
        phi_star=phi_star,
        h_f=h_f,
        h_dos=h_dos,
        fermi_ev=12.3,
        pressure_gpa=200.0,
        energy_ry=-50.0,
        grid_shape=(27, 27, 27),
        atoms=None,
    )


def stub_funnel(monkeypatch, outcome):
    """Replace run_ground_state with something that does not need QE."""
    calls = []

    def _fake(atoms, settings, runner, workdir, **kwargs):
        calls.append({"workdir": workdir, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(pipeline_module, "run_ground_state", _fake)
    monkeypatch.setattr(
        "mcts_framework.superhydride.qe.evaluator.run_ground_state", _fake
    )
    return calls


def make_evaluator(settings, tmp_path, **kwargs):
    options = {
        "pressure_gpa": 200.0,
        "cache_path": str(tmp_path / "qe_cache.csv"),
    }
    options.update(kwargs)
    return QuantumEspressoEvaluator(
        settings, QERunner(mpi_command="", ranks=1), str(tmp_path / "runs"), **options
    )


# --- Configuration refusals ----------------------------------------------


def test_relax_without_a_pressure_is_refused_before_any_compute(settings, tmp_path):
    """A pressure-stabilised hydride relaxed to 0 GPa is a different material."""
    with pytest.raises(ValueError, match="pressure_gpa"):
        QuantumEspressoEvaluator(
            settings, QERunner(), str(tmp_path), relax=True, pressure_gpa=None
        )


def test_no_relax_needs_no_pressure(settings, tmp_path):
    evaluator = QuantumEspressoEvaluator(
        settings, QERunner(), str(tmp_path), relax=False
    )
    assert evaluator.pressure_gpa is None


def test_funnel_refuses_a_structure_without_hydrogen(settings, tmp_path):
    from ase import Atoms

    atoms = Atoms("LaBe", positions=[(0, 0, 0), (2, 2, 2)], cell=[5, 5, 5], pbc=True)
    with pytest.raises(ValueError, match="no hydrogen"):
        run_ground_state(
            atoms, settings, QERunner(), str(tmp_path), relax=False
        )


def test_funnel_refuses_relax_without_a_pressure(settings, tmp_path, make_structure):
    with pytest.raises(ValueError, match="pressure_gpa"):
        run_ground_state(
            make_structure().atoms, settings, QERunner(), str(tmp_path), relax=True
        )


# --- Results and caching --------------------------------------------------


async def test_descriptors_reach_the_reward(settings, tmp_path, make_structure, monkeypatch):
    stub_funnel(monkeypatch, fake_result())
    evaluator = make_evaluator(settings, tmp_path)
    properties = await evaluator.evaluate(make_structure())
    assert properties["phi"] == pytest.approx(0.60)
    assert properties["phi_star"] == pytest.approx(0.70)
    assert properties["h_dos"] == pytest.approx(0.55)
    assert properties["formula"] == "BeLaH8"
    assert 0.0 < TcReward().compute_reward(properties) <= 1.0


async def test_each_candidate_gets_its_own_directory(
    settings, tmp_path, make_structure, monkeypatch
):
    calls = stub_funnel(monkeypatch, fake_result())
    evaluator = make_evaluator(settings, tmp_path)
    await evaluator.evaluate(make_structure("La", "Be"))
    await evaluator.evaluate(make_structure("Y", "Be"))
    assert len({call["workdir"] for call in calls}) == 2


async def test_relax_settings_are_passed_to_the_funnel(
    settings, tmp_path, make_structure, monkeypatch
):
    calls = stub_funnel(monkeypatch, fake_result())
    evaluator = make_evaluator(settings, tmp_path, relax=True, relax_passes=3)
    await evaluator.evaluate(make_structure())
    assert calls[0]["relax"] is True
    assert calls[0]["relax_passes"] == 3
    assert calls[0]["pressure_gpa"] == 200.0


async def test_results_are_written_to_the_cache(
    settings, tmp_path, make_structure, monkeypatch
):
    stub_funnel(monkeypatch, fake_result())
    cache = tmp_path / "qe_cache.csv"
    evaluator = make_evaluator(settings, tmp_path, cache_path=str(cache))
    await evaluator.evaluate(make_structure())

    frame = pd.read_csv(cache)
    assert len(frame) == 1
    assert frame.iloc[0]["formula"] == "BeLaH8"
    assert frame.iloc[0]["phi"] == pytest.approx(0.60)
    assert frame.iloc[0]["status"] == "ok"


async def test_an_interrupted_campaign_resumes_from_the_cache(
    settings, tmp_path, make_structure, monkeypatch
):
    stub_funnel(monkeypatch, fake_result())
    cache = tmp_path / "qe_cache.csv"
    await make_evaluator(settings, tmp_path, cache_path=str(cache)).evaluate(
        make_structure()
    )

    # A fresh evaluator whose funnel would explode must not need to call it.
    stub_funnel(monkeypatch, QEError("should not be reached"))
    resumed = make_evaluator(settings, tmp_path, cache_path=str(cache))
    properties = await resumed.evaluate(make_structure())
    assert properties["phi"] == pytest.approx(0.60)


async def test_a_finished_campaign_replays_as_a_descriptor_table(
    settings, tmp_path, make_structure, monkeypatch
):
    """The cache is written in the descriptor-table schema on purpose."""
    stub_funnel(monkeypatch, fake_result())
    cache = tmp_path / "qe_cache.csv"
    await make_evaluator(settings, tmp_path, cache_path=str(cache)).evaluate(
        make_structure()
    )

    table = DescriptorTableEvaluator(str(cache))
    replayed = await table.evaluate(make_structure())
    assert replayed["phi"] == pytest.approx(0.60)
    assert replayed["phi_star"] == pytest.approx(0.70)
    assert replayed["h_dos"] == pytest.approx(0.55)


# --- Failures -------------------------------------------------------------


async def test_a_failed_candidate_scores_zero_rather_than_killing_the_search(
    settings, tmp_path, make_structure, monkeypatch
):
    stub_funnel(monkeypatch, QEError("scf: convergence NOT achieved"))
    evaluator = make_evaluator(settings, tmp_path)
    properties = await evaluator.evaluate(make_structure())
    assert math.isnan(properties["phi"])
    assert properties["h_f"] == pytest.approx(0.8)  # still known from composition
    assert TcReward().compute_reward(properties) == 0.0


async def test_a_failure_is_recorded_so_it_is_not_retried(
    settings, tmp_path, make_structure, monkeypatch
):
    stub_funnel(monkeypatch, QEError("boom"))
    cache = tmp_path / "qe_cache.csv"
    evaluator = make_evaluator(settings, tmp_path, cache_path=str(cache))
    await evaluator.evaluate(make_structure())

    frame = pd.read_csv(cache)
    assert frame.iloc[0]["status"].startswith("failed")
    assert math.isnan(frame.iloc[0]["phi"])


async def test_one_failure_does_not_affect_the_next_candidate(
    settings, tmp_path, make_structure, monkeypatch
):
    evaluator = make_evaluator(settings, tmp_path)

    stub_funnel(monkeypatch, QEError("boom"))
    failed = await evaluator.evaluate(make_structure("La", "Be"))

    stub_funnel(monkeypatch, fake_result())
    succeeded = await evaluator.evaluate(make_structure("Y", "Be"))

    assert TcReward().compute_reward(failed) == 0.0
    assert TcReward().compute_reward(succeeded) > 0.0


def test_preflight_reports_missing_binaries(settings, tmp_path):
    evaluator = QuantumEspressoEvaluator(
        settings,
        QERunner(bin_dir=str(tmp_path / "nonexistent"), mpi_command=""),
        str(tmp_path),
        relax=False,
    )
    assert evaluator.preflight() == {"pw.x": False, "pp.x": False, "projwfc.x": False}


# --- Opt-in integration ---------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MCTS_QE_INTEGRATION"),
    reason="set MCTS_QE_INTEGRATION=1 (plus QE_BIN_DIR, QE_ENV_SETUP, "
    "ESPRESSO_PSEUDO) to run the real binaries",
)
def test_real_quantum_espresso_funnel(tmp_path):
    """
    The whole funnel against real binaries, on H3S Im-3m at ~200 GPa.

    Reduced cutoff so it finishes in minutes - a toolchain check, not a
    production protocol. Expect phi ~ 0.87 with a semicore-free sulphur
    pseudopotential (the published value is 0.68; see the descriptor pitfall).
    """
    from ase import Atoms

    a = 2.984
    atoms = Atoms(
        symbols=["S", "S"] + ["H"] * 6,
        scaled_positions=[
            (0, 0, 0), (0.5, 0.5, 0.5),
            (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
            (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
        ],
        cell=[a, a, a],
        pbc=True,
    )
    settings = QESettings(
        ecutwfc=60.0, ecutrho=240.0, pseudo_dir=os.environ["ESPRESSO_PSEUDO"]
    )
    runner = QERunner(
        bin_dir=os.environ.get("QE_BIN_DIR", ""),
        environment_setup=os.environ.get("QE_ENV_SETUP"),
        ranks=int(os.environ.get("MCTS_QE_RANKS", "4")),
        timeout_s=3000,
    )
    assert all(runner.check_available().values()), "QE binaries not reachable"

    result = run_ground_state(
        atoms, settings, runner, str(tmp_path / "h3s"), relax=False
    )
    assert 0.0 <= result.phi <= 1.0
    assert 0.0 <= result.phi_star <= 1.0
    assert result.h_f == pytest.approx(0.75)
    assert 0.0 < result.h_dos < 1.0
    assert result.pressure_gpa == pytest.approx(200.0, abs=25.0)
