"""
Config validation and CLI wiring for the Quantum ESPRESSO evaluator.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import pytest

pytest.importorskip("ase", reason="the superhydride builder needs ASE")
pytest.importorskip("spglib", reason="structure identifiers need spglib")

from ase.io import write as ase_write  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from mcts_framework.cli.builders import build_mcts  # noqa: E402
from mcts_framework.core.config import (  # noqa: E402
    Config,
    QuantumEspressoConfig,
    SuperhydrideConfig,
)
from mcts_framework.superhydride import DescriptorTableEvaluator  # noqa: E402
from mcts_framework.superhydride.qe import QuantumEspressoEvaluator  # noqa: E402


@pytest.fixture
def make_template(make_superhydride_template):
    return make_superhydride_template


@pytest.fixture
def template_cif(tmp_path, make_template):
    path = tmp_path / "template.cif"
    ase_write(str(path), make_template())
    return str(path)


@pytest.fixture
def pseudo_dir(tmp_path):
    directory = tmp_path / "pseudo"
    directory.mkdir()
    return str(directory)


def qe_section(pseudo_dir, **overrides):
    section = {"pseudo_dir": pseudo_dir, "pressure_gpa": 200.0, "work_root": "runs"}
    section.update(overrides)
    return section


def make_config(template_cif, superhydride=None):
    section = {"structure_path": template_cif}
    section.update(superhydride or {})
    return Config(
        material_type="superhydride",
        mcts={"iterations": 5},
        superhydride=section,
    )


# --- QuantumEspressoConfig ------------------------------------------------


def test_defaults_follow_the_protocol(pseudo_dir):
    qe = QuantumEspressoConfig(**qe_section(pseudo_dir))
    assert qe.ecutwfc == 90.0
    assert qe.ecutrho == 360.0        # ratio 4, right for norm-conserving
    assert qe.degauss == 0.02
    assert qe.relax is True
    assert qe.relax_passes == 2       # the Pulay error needs two
    assert qe.keep_cube is False      # cubes are tens of MB each


def test_relax_requires_a_target_pressure(pseudo_dir):
    with pytest.raises(ValidationError, match="pressure_gpa"):
        QuantumEspressoConfig(pseudo_dir=pseudo_dir, relax=True, pressure_gpa=None)


def test_no_relax_needs_no_pressure(pseudo_dir):
    assert QuantumEspressoConfig(pseudo_dir=pseudo_dir, relax=False).pressure_gpa is None


def test_a_pseudo_directory_is_required(monkeypatch):
    monkeypatch.delenv("ESPRESSO_PSEUDO", raising=False)
    with pytest.raises(ValidationError, match="pseudopotential directory"):
        QuantumEspressoConfig(relax=False)


def test_cluster_paths_fall_back_to_the_environment(monkeypatch, pseudo_dir):
    """They belong in the environment, not in a shareable config file."""
    monkeypatch.setenv("ESPRESSO_PSEUDO", pseudo_dir)
    monkeypatch.setenv("QE_BIN_DIR", "/opt/qe/bin")
    monkeypatch.setenv("QE_ENV_SETUP", "module load gcc/13.2.0 openmpi/4.1.6")

    qe = QuantumEspressoConfig(relax=False)
    assert qe.pseudo_dir == pseudo_dir
    assert qe.bin_dir == "/opt/qe/bin"
    assert "module load" in qe.environment_setup


def test_an_explicit_value_beats_the_environment(monkeypatch, pseudo_dir):
    monkeypatch.setenv("ESPRESSO_PSEUDO", "/wrong/path")
    assert QuantumEspressoConfig(pseudo_dir=pseudo_dir, relax=False).pseudo_dir == pseudo_dir


def test_unknown_field_is_rejected(pseudo_dir):
    with pytest.raises(ValidationError):
        QuantumEspressoConfig(**qe_section(pseudo_dir), ecut=90)


# --- Selecting the evaluator ---------------------------------------------


def test_table_is_the_default_evaluator():
    assert SuperhydrideConfig(structure_path="t.cif").evaluator == "table"


def test_quantum_espresso_requires_its_section():
    with pytest.raises(ValidationError, match="requires a 'quantum_espresso' section"):
        SuperhydrideConfig(structure_path="t.cif", evaluator="quantum_espresso")


def test_unknown_evaluator_is_rejected():
    with pytest.raises(ValidationError):
        SuperhydrideConfig(structure_path="t.cif", evaluator="vasp")


# --- Builder --------------------------------------------------------------


def test_builder_selects_the_table_evaluator_by_default(template_cif):
    mcts = build_mcts(make_config(template_cif))
    assert isinstance(mcts.property_evaluator, DescriptorTableEvaluator)


def test_builder_selects_the_qe_evaluator(template_cif, pseudo_dir, tmp_path):
    config = make_config(
        template_cif,
        {
            "evaluator": "quantum_espresso",
            "quantum_espresso": qe_section(
                pseudo_dir, work_root=str(tmp_path / "runs")
            ),
        },
    )
    evaluator = build_mcts(config).property_evaluator
    assert isinstance(evaluator, QuantumEspressoEvaluator)
    assert evaluator.pressure_gpa == 200.0
    assert evaluator.settings.pseudo_dir == pseudo_dir


def test_builder_passes_the_whole_protocol_through(template_cif, pseudo_dir, tmp_path):
    config = make_config(
        template_cif,
        {
            "evaluator": "quantum_espresso",
            "quantum_espresso": qe_section(
                pseudo_dir,
                work_root=str(tmp_path / "runs"),
                ecutwfc=70.0,
                ecutrho=700.0,
                degauss=0.03,
                kspacing_scf=0.3,
                kspacing_nscf=0.15,
                ranks=16,
                mpi_command="srun",
                bin_dir="/opt/qe/bin",
                environment_setup="module load gcc/13.2.0",
                relax_passes=3,
                pseudo_files={"H": "H_ONCV_PBE-1.0.oncvpsp.upf"},
            ),
        },
    )
    evaluator = build_mcts(config).property_evaluator
    assert evaluator.settings.ecutwfc == 70.0
    assert evaluator.settings.ecutrho == 700.0
    assert evaluator.settings.ecut_ratio() == 10.0   # a PAW ratio
    assert evaluator.settings.degauss == 0.03
    assert evaluator.settings.kspacing_scf == 0.3
    assert evaluator.settings.pseudo_files == {"H": "H_ONCV_PBE-1.0.oncvpsp.upf"}
    assert evaluator.runner.ranks == 16
    assert evaluator.runner.mpi_command == "srun"
    assert evaluator.runner.bin_dir == "/opt/qe/bin"
    assert evaluator.relax_passes == 3


def test_the_protocol_survives_a_yaml_round_trip(template_cif, pseudo_dir, tmp_path):
    """The config is dumped next to the results; a campaign must be replayable."""
    config = make_config(
        template_cif,
        {
            "evaluator": "quantum_espresso",
            "quantum_espresso": qe_section(pseudo_dir, ecutwfc=70.0, relax_passes=3),
        },
    )
    path = tmp_path / "config.yaml"
    config.dump_yaml(str(path))

    reloaded = Config.from_yaml(str(path))
    assert reloaded.superhydride.evaluator == "quantum_espresso"
    assert reloaded.superhydride.quantum_espresso.ecutwfc == 70.0
    assert reloaded.superhydride.quantum_espresso.relax_passes == 3
    assert reloaded.superhydride.quantum_espresso.pressure_gpa == 200.0
