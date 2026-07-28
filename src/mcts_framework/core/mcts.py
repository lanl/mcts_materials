"""
Core Monte Carlo Tree Search algorithm (material-agnostic).

The MCTS class orchestrates the four classic phases each iteration:

    selection -> expansion -> simulation -> backpropagation

It is completely generic over the material type M. All material-specific
behavior is injected via four collaborators:

    - MoveGenerator      : how to enumerate child materials (expansion)
    - PropertyEvaluator  : how to compute properties (simulation)
    - RewardFunction     : how to score properties (simulation)
    - SelectionStrategy  : how to descend the tree (selection)

© 2026. Triad National Security, LLC. All rights reserved.
"""

import logging
import random
from typing import TypeVar, Generic, List, Optional, Set, Dict, Any

import numpy as np

from .material import Material
from .move_generator import MoveGenerator
from .evaluator import PropertyEvaluator
from .reward import RewardFunction
from .search_node import SearchNode
from .selection import SelectionStrategy, AllChildrenTerminated

M = TypeVar('M', bound=Material)

logger = logging.getLogger(__name__)


class MCTS(Generic[M]):
    """
    Monte Carlo Tree Search over an abstract material space.

    Works with any material type that provides a MoveGenerator,
    PropertyEvaluator, RewardFunction, and SelectionStrategy.

    Deduplication semantics ("reserve only on attach"):
        A material is recorded in `visited_materials` only when it is actually
        attached to the tree as a child node. When a node first expands it
        generates its candidate moves and stores those not-yet-seen at that
        moment in `pending_children`, but does NOT reserve them. Because a
        different search path may attach one of those same materials before
        this node gets around to it, `_expand` re-checks each candidate at
        attach time and skips any that have since been claimed. This
        guarantees every material appears at most once in the tree while
        never "losing" a candidate that was generated but not yet attached.
    """

    def __init__(
        self,
        root_material: M,
        move_generator: MoveGenerator[M],
        property_evaluator: PropertyEvaluator,
        reward_function: RewardFunction,
        selection_strategy: SelectionStrategy[M],
        exploration_constant: float = 0.1,
        termination_limit: int = 60,
        rollout_depth: int = 1,
        n_rollout: int = 5,
        rollout_aggregation: str = "max",
        search_mode: str = "fast",
        seed: Optional[int] = None,
    ):
        """
        Args:
            root_material: Starting material at the tree root.
            move_generator: Enumerates child materials during expansion.
            property_evaluator: Computes (and caches) material properties.
            reward_function: Maps properties -> scalar reward.
            selection_strategy: Chooses a child during the selection walk.
            exploration_constant: UCB/PUCT exploration weight (c).
            termination_limit: Visits-without-improvement before a node
                is marked terminated.
            rollout_depth: Number of random substitution steps per rollout
                sample beyond the node itself (depth 0 = evaluate the node).
            n_rollout: Total rollout samples per newly expanded node,
                including the mandatory depth-0 sample.
            rollout_aggregation: How to combine a node's n_rollout reward
                samples into its value:
                - 'max' (default): optimistic maximum. The extra (depth>0)
                  samples are discounted by 0.9**rollout_depth before
                  comparison, acting as a confidence penalty on speculative
                  lookahead relative to the node's own depth-0 reward.
                - 'mean': plain average of UNDISCOUNTED samples, an unbiased
                  estimate of expected reward. The depth discount is dropped
                  here (discounting-then-averaging-by-unweighted-n would just
                  drag the mean toward zero rather than weight confidence).
            search_mode: When the run stops.
                - 'fast' (default): stop as soon as the ROOT node self-
                  terminates via its visits-without-improvement countdown
                  (i.e. the search has converged). Minimizes evaluations
                  (DFT/MACE calls); best when finding the single optimum
                  quickly matters most.
                - 'thorough': ignore the root's countdown self-termination as
                  a stop signal, so the run continues to the full `iterations`
                  budget. It stops earlier only on TRUE EXHAUSTION - when
                  terminations cascade up to the root, i.e. the root becomes
                  fully expanded AND all its children are terminated (detected
                  in _select, which raises AllChildrenTerminated). Because a
                  node with unexpanded children is never structurally
                  terminated, exhaustion means the reachable space has been
                  fully attached. Explores more compounds for a better top-N
                  list, at the cost of more evaluations.
                In BOTH modes a node's countdown can still flag it terminated
                and selection always skips terminated children; the mode only
                governs whether the root's countdown-termination *ends the run*.
            seed: Optional RNG seed for reproducibility.
        """
        if rollout_aggregation not in ("max", "mean"):
            raise ValueError(
                f"rollout_aggregation must be 'max' or 'mean', "
                f"got {rollout_aggregation!r}"
            )
        if search_mode not in ("fast", "thorough"):
            raise ValueError(
                f"search_mode must be 'fast' or 'thorough', got {search_mode!r}"
            )
        self.exploration_constant = exploration_constant
        self.termination_limit = termination_limit
        self.rollout_depth = rollout_depth
        self.n_rollout = max(1, n_rollout)
        self.rollout_aggregation = rollout_aggregation
        self.search_mode = search_mode

        self.root: SearchNode[M] = SearchNode(
            root_material,
            exploration_constant=exploration_constant,
            termination_limit=termination_limit,
        )

        self.move_generator = move_generator
        self.property_evaluator = property_evaluator
        self.reward_function = reward_function
        self.selection_strategy = selection_strategy

        # Dedicated RNG so runs are reproducible and independent of global state.
        self._rng = random.Random(seed)

        # Global search state.
        self.terminated: bool = False
        self.best_node: Optional[SearchNode[M]] = None
        self.best_reward: float = -np.inf
        self.iteration: int = 0

        # Identifiers of materials ATTACHED to the tree (reserve-on-attach).
        self.visited_materials: Set[str] = set()
        self.visited_materials.add(root_material.get_identifier())

        # Per-iteration history for convergence analysis.
        self.reward_history: List[float] = []
        self.unique_materials_history: List[int] = []

    # ------------------------------------------------------------------ #
    # Public driver
    # ------------------------------------------------------------------ #

    async def run(self, iterations: int) -> None:
        """
        Run the search for up to `iterations` iterations.

        Stopping depends on search_mode:
        - 'fast': stop on true exhaustion (self.terminated) OR when the root
          self-terminates via its no-improvement countdown (root.terminated).
        - 'thorough': stop only on true exhaustion; the root's countdown
          termination is ignored so the full iteration budget is used.
        """
        logger.info("Starting MCTS: %d iterations (mode=%s)",
                    iterations, self.search_mode)

        for i in range(iterations):
            self.iteration = i

            converged = self.root.terminated and self.search_mode == "fast"
            if self.terminated or converged:
                logger.info("Stopping at iteration %d (%s)", i,
                            "exhausted" if self.terminated else "converged")
                break

            try:
                await self._step()
            except Exception as exc:  # keep the search alive on a bad node
                logger.error("Iteration %d failed: %s", i, exc)

            self.reward_history.append(self.best_reward)
            self.unique_materials_history.append(len(self.visited_materials))

        logger.info(
            "MCTS complete. Best reward=%.4f, unique materials=%d",
            self.best_reward,
            len(self.visited_materials),
        )

    async def _step(self) -> None:
        """Run one selection -> expansion -> simulation -> backprop cycle."""
        # 1. Selection: walk to a node we can expand (or a leaf).
        node, chain = self._select()

        if node is None:
            # Selection found every branch terminated.
            self.terminated = True
            return

        # 2. Expansion: attach one new child if this node still expands.
        target = node
        if node.expandable:
            child = await self._expand(node)
            if child is not None:
                target = child
                chain = chain + [child]

        # 3. Simulation: evaluate the target node. Returns the value to
        #    backpropagate (max over the node's own eval and any discounted
        #    rollout samples); target.own_reward is the node's OWN value.
        backprop_value = await self._simulate(target)

        # 4. Backpropagation: push the (lookahead) value up the chain. This
        #    feeds selection and the subtree_best termination heuristic.
        self._backpropagate(chain, backprop_value)

        # 5. Track the global best by the node's OWN reward, so the reported
        #    champion material and its reward always agree (rollout samples of
        #    other materials must not define the champion).
        own = target.own_reward
        if own is not None and own > self.best_reward:
            self.best_reward = own
            self.best_node = target

    # ------------------------------------------------------------------ #
    # Phase 1: Selection
    # ------------------------------------------------------------------ #

    def _select(self):
        """
        Walk from root using the selection strategy until we reach a node
        that is still expandable or a leaf.

        Returns:
            (node, chain) where chain is the list of nodes from root to node
            (inclusive). node is None if every branch is terminated.

        If a fully-expanded node is found to have only terminated children, it
        is itself marked terminated and the walk restarts from the root. This
        is done with an outer loop (not recursion) so a deep, mostly-terminated
        tree cannot exhaust the Python recursion limit.
        """
        while True:
            current = self.root
            chain: List[SearchNode[M]] = [current]

            dead_branch = False
            # Descend while the node is fully expanded and has children.
            while current.is_fully_expanded() and not current.is_leaf():
                try:
                    current = self.selection_strategy.select_child(current)
                except AllChildrenTerminated:
                    current.terminated = True
                    if current is self.root:
                        return None, chain
                    dead_branch = True
                    break  # restart the walk from the root
                chain.append(current)

            if not dead_branch:
                return current, chain

    # ------------------------------------------------------------------ #
    # Phase 2: Expansion
    # ------------------------------------------------------------------ #

    async def _expand(self, node: SearchNode[M]) -> Optional[SearchNode[M]]:
        """
        Add a single new child to `node`.

        On first expansion, enumerate all moves, dedupe within the candidate
        set, drop the parent itself, and drop anything already attached to the
        tree AT THAT MOMENT. Candidates are stored in `pending_children` but
        NOT reserved.

        On every expansion, pop candidates until one is found that is still
        unclaimed (not in `visited_materials`) - another search path may have
        attached it in the meantime. Only when a child is actually attached is
        its identifier reserved. This never loses a candidate: a material stays
        poppable until some node attaches it.
        """
        if not node.pending_children and node.expandable:
            candidates = self.move_generator.generate_moves(node.material)
            candidates = self.move_generator.filter_invalid(candidates)

            # Dedupe within this candidate set, skip the parent, and skip
            # anything already attached to the tree at generation time.
            seen: Set[str] = set()
            fresh: List[M] = []
            parent_id = node.material.get_identifier()
            for mat in candidates:
                ident = mat.get_identifier()
                if ident == parent_id or ident in seen:
                    continue
                if ident in self.visited_materials:
                    continue
                seen.add(ident)
                fresh.append(mat)

            node.pending_children = fresh

        # Pop until we find a still-unclaimed material or run out. A candidate
        # generated earlier may have been attached elsewhere since; if so, skip
        # it (it now lives in the tree under its first owner).
        while node.pending_children:
            child_material = node.pending_children.pop()
            ident = child_material.get_identifier()
            if ident in self.visited_materials:
                continue  # claimed by another path since generation - skip

            child = SearchNode(
                child_material,
                exploration_constant=self.exploration_constant,
                termination_limit=self.termination_limit,
            )
            node.add_child(child)
            self.visited_materials.add(ident)  # reserve only on attach

            if not node.pending_children:
                node.expandable = False
            return child

        # Nothing left to attach.
        node.expandable = False
        return None

    # ------------------------------------------------------------------ #
    # Phase 3: Simulation (rollout)
    # ------------------------------------------------------------------ #

    async def _simulate(self, node: SearchNode[M]) -> float:
        """
        Estimate the value of `node` by aggregating n_rollout reward samples.

        Sample 0 always evaluates the node itself (depth 0), records its
        properties, and sets node.own_reward (the node's true material value,
        which stays independent of aggregation). The remaining n_rollout-1
        samples are random "max-along-walk" rollouts (see _rollout_sample).

        The samples are combined per self.rollout_aggregation:
        - 'max': the extra samples are discounted by 0.9**rollout_depth (a
          confidence penalty on speculative lookahead), then the maximum over
          all samples is taken.
        - 'mean': the extra samples are left undiscounted and the plain mean
          over all samples is returned (unbiased expected-reward estimate).

        The returned value is what backpropagation propagates; node.own_reward
        is always the undiscounted depth-0 reward regardless of aggregation.
        """
        # Depth-0 sample: evaluate the node itself and cache its properties.
        base_props = await self.property_evaluator.evaluate(node.material)
        node.properties = base_props
        own = self.reward_function.compute_reward(base_props)
        node.own_reward = own

        samples = [own]

        # 'max' discounts the extra samples; 'mean' leaves them undiscounted.
        scale = 0.9 ** self.rollout_depth if self.rollout_aggregation == "max" else 1.0
        for _ in range(self.n_rollout - 1):
            samples.append(scale * await self._rollout_sample(node.material))

        if self.rollout_aggregation == "max":
            return max(samples)
        return sum(samples) / len(samples)  # 'mean'

    async def _rollout_sample(self, material: M) -> float:
        """
        One "max-along-walk" random rollout.

        Take up to rollout_depth independent random moves and evaluate the
        reward at EVERY intermediate composition, returning the maximum seen
        along the walk. Every composition the walk passes through is a valid
        candidate, so scoring only the endpoint would discard information;
        max-along-walk extracts up to `rollout_depth` candidate evaluations
        per walk instead of one. Returns the node's own reward if no move is
        possible.

        Rollout samples are NOT added to the tree and do NOT reserve
        identifiers - they only probe reward, so their materials remain
        available for real expansion later. The depth discount (for 'max'
        aggregation) is applied by the caller, not here.
        """
        current = material
        step_rewards: List[float] = []

        for _ in range(self.rollout_depth):
            moves = self.move_generator.generate_moves(current)
            moves = self.move_generator.filter_invalid(moves)
            if not moves:
                break
            current = self._rng.choice(moves)
            props = await self.property_evaluator.evaluate(current)
            step_rewards.append(self.reward_function.compute_reward(props))

        if not step_rewards:
            # No move was possible; fall back to the starting material's reward.
            props = await self.property_evaluator.evaluate(material)
            return self.reward_function.compute_reward(props)

        return max(step_rewards)

    # ------------------------------------------------------------------ #
    # Phase 4: Backpropagation
    # ------------------------------------------------------------------ #

    def _backpropagate(self, chain: List[SearchNode[M]], reward: float) -> None:
        """Update every node on the selection chain with the reward."""
        for node in chain:
            node.update(reward)

    # ------------------------------------------------------------------ #
    # Results / inspection helpers
    # ------------------------------------------------------------------ #

    def all_nodes(self) -> List[SearchNode[M]]:
        """Return every node in the tree (iterative DFS)."""
        out: List[SearchNode[M]] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out

    def to_tree_dict(self) -> Dict[str, Any]:
        """
        Serialize the explored tree to a portable, JSON-safe dict.

        Captures structure (parent links) plus each node's identifier and MCTS
        statistics (own_reward, visits, total_reward, subtree_best) and its
        evaluated properties - enough to redraw the search tree offline without
        re-running the search or unpickling live objects (ASE Atoms, evaluator
        handles). Node ids are indices into a DFS ordering; the root is id 0.
        """
        nodes = self.all_nodes()
        index = {id(n): i for i, n in enumerate(nodes)}
        records = []
        for i, n in enumerate(nodes):
            records.append({
                "id": i,
                "parent": index[id(n.parent)] if n.parent is not None else None,
                "identifier": n.material.get_identifier(),
                "own_reward": n.own_reward,
                "visits": n.visits,
                "total_reward": n.total_reward,
                "subtree_best": (
                    None if n.subtree_best == float("-inf") else n.subtree_best
                ),
                "terminated": n.terminated,
                "properties": dict(n.properties),
            })
        return {"root_id": 0, "nodes": records}

    def save_tree_json(self, path: str) -> None:
        """Write to_tree_dict() to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_tree_dict(), f, indent=2)

    def get_best_materials(self, n: int = 10) -> List[SearchNode[M]]:
        """
        Return the top-n nodes ranked by their OWN evaluated reward.

        Ranking uses own_reward (each node's own material value), not
        best_reward - the latter is a subtree maximum accumulated during
        backpropagation, so internal nodes would otherwise inherit the score
        of their best descendant and outrank genuine leaf candidates. Only
        nodes that have actually been evaluated (own_reward is set) are
        included.
        """
        evaluated = [nd for nd in self.all_nodes() if nd.own_reward is not None]
        evaluated.sort(key=lambda nd: nd.own_reward, reverse=True)
        return evaluated[:n]

    def summary(self) -> Dict[str, Any]:
        """Return a small dict summarizing the completed search."""
        return {
            "iterations": self.iteration + 1,
            "best_reward": self.best_reward,
            "best_material": (
                self.best_node.material.get_identifier()
                if self.best_node is not None
                else None
            ),
            "unique_materials": len(self.visited_materials),
            "tree_size": self.root.get_subtree_size(),
        }
