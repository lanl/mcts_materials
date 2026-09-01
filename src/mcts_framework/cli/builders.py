"""
Build a ready-to-run MCTS instance from a validated Config.

Dispatches on Config.material_type to assemble the right material, move
generator, property evaluator, and reward function, then wires them into the
generic MCTS with the configured selection strategy and hyperparameters.

Material-specific imports (ase/rdkit/etc.) are done lazily inside each builder
so importing this module - and the CLI - stays lightweight.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Tuple, TYPE_CHECKING

from ..core.config import Config
from ..core.mcts import MCTS
from ..core.selection import create_selection_strategy

if TYPE_CHECKING:
    from ..core.evaluator import PropertyEvaluator
    from ..core.reward import RewardFunction


def build_mcts(config: Config) -> MCTS:
    """
    Construct an MCTS instance from a validated Config.

    Args:
        config: A fully-validated top-level Config.

    Returns:
        An MCTS ready to `await run(...)`.

    Raises:
        ValueError: on an unsupported material_type.
    """
    if config.material_type == "intermetallic":
        root, moves, evaluator, reward = _build_intermetallic(config)
    elif config.material_type == "molecule":
        root, moves, evaluator, reward = _build_molecule(config)
    elif config.material_type == "superhydride":
        root, moves, evaluator, reward = _build_superhydride(config)
    else:  # pragma: no cover - Config validation prevents this
        raise ValueError(f"Unsupported material_type: {config.material_type!r}")

    selection = create_selection_strategy(
        config.mcts.selection_mode,
        epsilon=config.mcts.epsilon,
        temperature=config.mcts.temperature,
    )

    return MCTS(
        root_material=root,
        move_generator=moves,
        property_evaluator=evaluator,
        reward_function=reward,
        selection_strategy=selection,
        exploration_constant=config.mcts.exploration_constant,
        termination_limit=config.mcts.termination_limit,
        rollout_depth=config.mcts.rollout_depth,
        n_rollout=config.mcts.n_rollout,
        rollout_aggregation=config.mcts.rollout_aggregation,
        search_mode=config.mcts.search_mode,
        seed=config.mcts.seed,
    )


def _build_intermetallic(config: Config) -> Tuple[object, object, "PropertyEvaluator", "RewardFunction"]:
    """Assemble intermetallic components from config.intermetallic."""
    from ase.io import read
    from ase.data import atomic_numbers
    from ..intermetallic import (
        IntermetallicStructure,
        PeriodicTableMoves,
        MaceEvaluator,
        DoscarRewardLookup,
        create_intermetallic_reward,
        elements,
    )

    ic = config.intermetallic
    atoms = read(ic.structure_path)

    # DEPRECATED: composition override mechanism. Kept for backward compatibility.
    # Users should now specify the desired starting composition directly in the CIF
    # file via structure_path. The config validator checks that the CIF's f-block
    # element(s) are compatible with f_block_mode.
    #
    # Apply composition overrides to the root structure before search starts.
    # If transition_metal or group_iv are specified, substitute those elements
    # in the loaded structure so the search starts from the desired composition.
    if ic.transition_metal or ic.group_iv:
        target_tm_z = atomic_numbers.get(ic.transition_metal, 0) if ic.transition_metal else 0
        target_giv_z = atomic_numbers.get(ic.group_iv, 0) if ic.group_iv else 0

        op = []
        for z in atoms.get_atomic_numbers():
            z = int(z)
            # f-block sites: leave unchanged (will be substituted by MCTS)
            if z in elements.F_BLOCK_ELEMENTS:
                op.append(0)
            # Group IV sites: substitute if specified
            elif z in elements.GROUP_IV_CHAIN:
                op.append((target_giv_z - z) if target_giv_z else 0)
            # Metal sites: substitute if specified
            else:
                op.append((target_tm_z - z) if target_tm_z else 0)

        atoms = atoms.copy()
        atoms.set_atomic_numbers(atoms.get_atomic_numbers() + op)

    # Validate the (final) starting composition against f_block_mode, now that
    # the CIF is loaded. Done here rather than in config validation so the check
    # runs on the atoms we already have - no extra file I/O, no ASE dependency in
    # the config layer. Raises on a hard incompatibility (e.g. u_only without U);
    # softer issues are surfaced as warnings.
    import warnings

    for msg in elements.validate_fblock_compat(atoms.get_atomic_numbers(), ic.f_block_mode):
        warnings.warn(msg, UserWarning)

    root = IntermetallicStructure(atoms)

    moves = PeriodicTableMoves(
        f_block_mode=ic.f_block_mode, move_step=ic.move_step, u_bridge=ic.u_bridge
    )
    evaluator = MaceEvaluator(cache_path=ic.cache_path, mp_api_key=ic.mp_api_key)

    # DOSCAR lookup needed for the rDOS-using methods.
    doscar = None
    if ic.rollout_method in ("rdos", "ehull_rdos", "ehull_rdos_product"):
        doscar = DoscarRewardLookup(peaks_file=ic.doscar_data_path)

    reward = create_intermetallic_reward(
        ic.rollout_method,
        doscar_lookup=doscar,
        beta=ic.beta,
        gamma=ic.gamma,
    )
    return root, moves, evaluator, reward


def _build_superhydride(config: Config) -> Tuple[object, object, "PropertyEvaluator", "RewardFunction"]:
    """Assemble superhydride components from config.superhydride."""
    import warnings

    from ase.io import read

    from ..superhydride import (
        DescriptorTableEvaluator,
        HostSubstitutionMoves,
        SuperhydrideStructure,
        create_superhydride_reward,
        elements,
    )

    sc = config.superhydride
    atoms = read(sc.structure_path)

    # Validate the template against the palette now that the CIF is loaded.
    # Raises if the structure is not a hydride or has no movable host site;
    # softer problems (a frozen host, a binary template) come back as warnings.
    for msg in elements.validate_hosts(atoms.get_atomic_numbers(), sc.host_palette):
        warnings.warn(msg, UserWarning)

    root = SuperhydrideStructure(atoms)
    moves = HostSubstitutionMoves(
        palette=sc.host_palette,
        preserve_distinct_hosts=sc.preserve_distinct_hosts,
    )

    if sc.evaluator == "quantum_espresso":
        evaluator = _build_qe_evaluator(sc.quantum_espresso)
    else:
        evaluator = DescriptorTableEvaluator(table_path=sc.descriptor_table_path)

    reward = create_superhydride_reward(normalize=sc.normalize_reward)
    return root, moves, evaluator, reward


def _build_qe_evaluator(qc: object) -> "PropertyEvaluator":
    """Assemble the Quantum ESPRESSO evaluator from config.superhydride.quantum_espresso."""
    from ..superhydride.qe import QERunner, QESettings, QuantumEspressoEvaluator

    settings = QESettings(
        ecutwfc=qc.ecutwfc,
        ecutrho=qc.ecutrho,
        degauss=qc.degauss,
        conv_thr=qc.conv_thr,
        kspacing_scf=qc.kspacing_scf,
        kspacing_nscf=qc.kspacing_nscf,
        pseudo_dir=qc.pseudo_dir,
        pseudo_files=dict(qc.pseudo_files),
    )
    runner = QERunner(
        bin_dir=qc.bin_dir or "",
        mpi_command=qc.mpi_command,
        ranks=qc.ranks,
        environment_setup=qc.environment_setup,
        timeout_s=qc.timeout_s,
    )
    return QuantumEspressoEvaluator(
        settings,
        runner,
        work_root=qc.work_root,
        pressure_gpa=qc.pressure_gpa,
        relax=qc.relax,
        relax_passes=qc.relax_passes,
        cache_path=qc.cache_path,
        keep_scratch=qc.keep_scratch,
        keep_cube=qc.keep_cube,
    )


def _build_molecule(config: Config) -> Tuple[object, object, "PropertyEvaluator", "RewardFunction"]:
    """Assemble molecule components from config.molecule."""
    from ..molecule import (
        MolecularStructure,
        FunctionalGroupMoves,
        MoleculeEvaluator,
        create_molecule_reward,
    )

    mc = config.molecule
    root = MolecularStructure.from_smiles(mc.starting_smiles)
    moves = FunctionalGroupMoves(functional_groups=mc.functional_groups)

    # Predict whatever properties the reward needs (plus melting_point default).
    reward = create_molecule_reward(mc.objective, weights=mc.objective_weights)
    properties = reward.get_property_names() or ["melting_point"]
    evaluator = MoleculeEvaluator(
        properties=properties,
        chemprop_model_dir=mc.chemprop_model_dir,
        xgboost_model_dir=mc.xgboost_model_dir,
    )
    return root, moves, evaluator, reward
