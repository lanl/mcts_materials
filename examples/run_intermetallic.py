"""
Example: intermetallic crystal-structure search from Python (no CLI).

Mirrors examples/config_intermetallic.yaml but wires the components by hand, so
you can see the programmatic API. Uses rollout_method='rdos' by default, which
needs only ASE + spglib and a DOSCAR peaks CSV - NO Materials Project API key.

Requires the [intermetallic] extra:
    pip install -e ".[intermetallic]"

And a DOSCAR peaks CSV (not bundled - see the main project's Data Availability).
Point DOSCAR_PATH below at your copy.

Run:  python examples/run_intermetallic.py

© 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio
from pathlib import Path

from ase.io import read

from mcts_framework import MCTS, UCB1
from mcts_framework.intermetallic import (
    IntermetallicStructure,
    PeriodicTableMoves,
    DoscarRewardLookup,
    RdosReward,
)

HERE = Path(__file__).parent
CIF_PATH = HERE / "mat_Pb6U1W6_sg191.cif"
DOSCAR_PATH = "doscar_peaks_data_with_U.csv"  # <-- set to your DOSCAR peaks CSV


async def main() -> None:
    atoms = read(str(CIF_PATH))
    root = IntermetallicStructure(atoms)

    # rDOS-only reward: no MACE / Materials Project needed. For an evaluator,
    # rdos reads properties["formula"], which the structure exposes via its
    # identifier; here we use a tiny evaluator that just records the formula.
    from mcts_framework.core.evaluator import PropertyEvaluator

    class FormulaEvaluator(PropertyEvaluator):
        """Minimal evaluator for rdos-only runs: records the formula only."""

        async def _compute(self, material):
            return {"formula": material.get_formula()}

    doscar = DoscarRewardLookup(peaks_file=DOSCAR_PATH)

    mcts = MCTS(
        root_material=root,
        move_generator=PeriodicTableMoves(f_block_mode="u_only"),
        property_evaluator=FormulaEvaluator(),
        reward_function=RdosReward(doscar),
        selection_strategy=UCB1(),
        exploration_constant=0.1,
        n_rollout=5,
        rollout_depth=1,
        seed=0,
    )

    await mcts.run(iterations=200)

    print("Summary:", mcts.summary())
    print("\nTop 5 by rDOS reward:")
    for node in mcts.get_best_materials(n=5):
        print(f"  {node.material.get_formula():<12} "
              f"reward={node.own_reward:.4g}  visits={node.visits}")


if __name__ == "__main__":
    asyncio.run(main())
