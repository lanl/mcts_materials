"""
Example: molecular search from Python (no CLI).

Mirrors examples/config_molecule.yaml but wires the components by hand.

Requires the [molecule] extra (RDKit) plus the molecule-modifier package and
its external model files for property prediction:
    pip install -e ".[molecule]"
    # then install molecule-modifier and set CHEMPROP_MODEL_DIR / XGBOOST_MODEL_DIR
    # (see that package's README).

Run:  python examples/run_molecule.py

© 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio

from mcts_framework import MCTS, UCB1
from mcts_framework.molecule import (
    MolecularStructure,
    FunctionalGroupMoves,
    MoleculeEvaluator,
    MeltingPointReward,
)


async def main() -> None:
    # Start from ethanol and grow it by attaching functional groups.
    root = MolecularStructure.from_smiles("CCO")

    moves = FunctionalGroupMoves(functional_groups=["C", "CC", "O", "N"])
    evaluator = MoleculeEvaluator(properties=["melting_point"])
    reward = MeltingPointReward(min_temp=200.0, max_temp=700.0)  # favors HIGH mp

    mcts = MCTS(
        root_material=root,
        move_generator=moves,
        property_evaluator=evaluator,
        reward_function=reward,
        selection_strategy=UCB1(),
        exploration_constant=0.1,
        n_rollout=5,
        rollout_depth=1,
        seed=0,
    )

    await mcts.run(iterations=200)

    print("Summary:", mcts.summary())
    print("\nTop 5 by melting-point reward:")
    for node in mcts.get_best_materials(n=5):
        mp = node.properties.get("melting_point")
        mp_str = f"{mp:.1f} K" if mp is not None else "n/a"
        print(f"  {node.material.get_smiles():<20} "
              f"reward={node.own_reward:.3f}  melting_point={mp_str}")


if __name__ == "__main__":
    asyncio.run(main())
