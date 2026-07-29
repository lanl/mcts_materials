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
