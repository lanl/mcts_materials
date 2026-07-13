"""
Monte Carlo Tree Search implementation for crystal structure optimization.
"""

import logging
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
from .node import MCTSTreeNode

logger = logging.getLogger(__name__)

# stat_dict entries gained a 6th element (dos_reward) after rDOS support was added;
# older runs/pickles may still have only the original 5.
_STAT_DICT_COLUMNS = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
_STAT_DICT_COLUMNS_WITH_DOS = _STAT_DICT_COLUMNS + ['dos_reward']


def stat_dict_to_dataframe(stat_dict: Dict) -> pd.DataFrame:
    """
    Convert an MCTS stat_dict (formula -> [best_reward, visit_count, terminated,
    e_above_hull, e_form, (dos_reward)]) into a DataFrame with those column names,
    indexed by formula. Handles both the 5-element (no dos_reward) and 6-element
    stat_dict formats, and an empty stat_dict (returned as an empty DataFrame,
    unchanged, since there are no columns to rename).
    """
    df = pd.DataFrame(stat_dict).T
    if df.empty:
        return df
    df.columns = _STAT_DICT_COLUMNS_WITH_DOS if df.shape[1] == 6 else _STAT_DICT_COLUMNS
    return df


class MCTS:
    """
    Monte Carlo Tree Search algorithm for crystal structure optimization.
    """
    
    def __init__(self, root: MCTSTreeNode, epsilon: float = 0.2, temperature: float = 1.0):
        """
        Initialize MCTS algorithm.

        Args:
            root: Root node of the MCTS tree
            epsilon: Exploration rate for epsilon-greedy/hybrid selection (default: 0.2)
            temperature: Temperature for 'boltzmann' selection mode (default: 1.0).
                Higher = more uniform/exploratory, lower = closer to greedy argmax.
                Unused by the other selection modes.
        """
        self.root = root
        self.origin_root = root
        self.current_node = root
        self.stat_dict: Dict[str, List] = {}
        self.t_warmup = len(root.metal_move) * len(root.g_iv_move)
        self.max_reward = -10.0
        self.max_reward_history = []  # max_reward (best composite reward so far) per iteration
        self.best_node: Optional[MCTSTreeNode] = None
        self.terminated = False
        self.epsilon = epsilon
        self.temperature = temperature

        # Track node with best E_form and its properties
        self.best_e_form_node: Optional[MCTSTreeNode] = None
        self.best_e_form_history = []
        self.best_e_form_e_hull_history = []
        self.best_e_form_formula_history = []

        # Track node with best E_hull and its properties
        self.best_e_hull_node: Optional[MCTSTreeNode] = None
        self.best_e_hull_history = []
        self.best_e_hull_e_form_history = []
        self.best_e_hull_formula_history = []

        # Track node with best rDOS and its properties
        self.best_rdos_node: Optional[MCTSTreeNode] = None
        self.best_rdos_history = []
        self.best_rdos_eform_history = []
        self.best_rdos_ehull_history = []
        self.best_rdos_formula_history = []
        self.best_rdos_value = 0.0  # Track best rDOS value seen so far

        # Track number of unique compounds discovered
        self.n_unique_compounds_history = []
        
    def select_node(self, mode: str = 'ucb1') -> List[MCTSTreeNode]:
        """
        Node selection algorithm: walk from the root to a leaf, picking one
        child at each level according to `mode`.

        Args:
            mode: Child-selection strategy. One of:
                - 'ucb1' (default): Classic UCB1 - always pick the child with
                  the highest UCB1 = Q/N + c*sqrt(ln(N_parent)/N). Deterministic;
                  the exploration/exploitation tradeoff comes entirely from the
                  formula itself, with no extra randomization layered on top.
                - 'epsilon_greedy': Textbook epsilon-greedy - argmax(UCB1) with
                  probability (1 - epsilon); with probability epsilon, pick a
                  child uniformly at random (ignoring UCB1 magnitude).
                - 'boltzmann': Stochastic at every step - P(child) is
                  proportional to exp(UCB1 / temperature) (softmax exploration).
                - 'puct': AlphaZero-style PUCT = Q + c*prior*sqrt(N_parent)/(1+N),
                  with a uniform prior (1/num_children) since there is no
                  learned policy network here. Deterministic argmax, like
                  'ucb1', but an unvisited child has Q=0 rather than +inf.
                - 'hybrid': This codebase's original default (previously named
                  'epsilon') - an epsilon-greedy outer loop, but the
                  exploratory branch draws from a roulette wheel weighted by
                  UCB1-squared rather than picking uniformly. Not a standard
                  textbook method; kept for backward compatibility/comparison.

        Returns:
            List of selected nodes for back-propagation
        """
        select_chain = [self.root]
        current = self.root

        while not current.expandable:
            n_children = len(current.children)
            prior = 1.0 / n_children if n_children > 0 else 0.0

            scores = []
            for child_node in current.children:
                if child_node.terminated:
                    scores.append(-1e4)
                    if child_node.get_chemical_formula() in self.stat_dict:
                        self.stat_dict[child_node.get_chemical_formula()][2] = True
                elif mode == 'puct':
                    scores.append(child_node.get_puct(prior))
                else:
                    scores.append(child_node.get_ucb())

            # Check if all children are terminated
            if set(scores) == {-1e4}:
                self.terminated = True
                break

            # Select next node based on mode
            if mode in ('ucb1', 'puct'):
                current = current.children[np.argmax(scores)]
            elif mode == 'epsilon_greedy':
                if random.random() < self.epsilon:
                    current = current.children[self._uniform_selector(scores)]
                else:
                    current = current.children[np.argmax(scores)]
            elif mode == 'boltzmann':
                current = current.children[self._boltzmann_selector(scores)]
            elif mode == 'hybrid':
                if random.random() < self.epsilon:
                    current = current.children[self._probability_selector(scores)]
                else:
                    current = current.children[np.argmax(scores)]
            else:
                raise ValueError(f"Unknown selection mode: {mode}")

            select_chain.append(current)

        self.current_node = current
        return select_chain
        
    def back_propagation(self, reward: float, select_chain: List[MCTSTreeNode], 
                        renew_t_to_terminate: bool):
        """
        Back-propagate rewards through the selection chain.
        
        Args:
            reward: Reward value to propagate
            select_chain: Chain of nodes to update
            renew_t_to_terminate: Whether to reset termination countdown
        """
        for node in select_chain:
            node.update_rewards(reward)
            node.visit(renew_t_to_terminate)
            
    def _run_rollout_samples(self, new_node: MCTSTreeNode, rollout_depth: int,
                              n_rollout: int, energy_calculator, mode: str,
                              doscar_lookup, n_workers: int,
                              rollout_aggregation: str = 'max',
                              rollout_discount: float = 0.9) -> List[float]:
        """
        Evaluate n_rollout independent rollout samples for new_node and return
        their (scaled) rewards. The first sample always uses depth=0 (evaluates
        new_node itself, no extra random substitution) and runs sequentially,
        since it has the side effect of recording new_node.e_form/e_above_hull.
        The remaining n_rollout - 1 samples are independent random walks of
        rollout_depth steps; when n_workers > 1 they are evaluated concurrently
        via a thread pool, each with its own deterministically-seeded RNG so
        results stay reproducible under --seed regardless of thread scheduling.

        The depth-dependent discount (0.9 ** rollout_depth) applied to those
        extra samples only makes sense under 'max' aggregation, where it acts
        as a confidence penalty on a speculative sample before comparing it
        against the node's own real (depth=0) reward. Under 'mean' aggregation
        the same multiply-then-average-with-unweighted-n approach would just
        drag the average toward zero by an amount that grows with n_rollout
        and rollout_depth, which isn't a meaningful confidence weighting - so
        for 'mean' the extra samples are left undiscounted (scale=1.0).

        Args:
            new_node: Node to roll out from
            rollout_depth: Depth of each additional rollout simulation
            n_rollout: Total number of rollout samples (including the depth=0 one)
            energy_calculator: Energy calculator instance
            mode: Rollout mode string passed through to MCTSTreeNode.rollout()
            doscar_lookup: DoscarRewardLookup instance for DOSCAR rewards
            n_workers: Number of worker threads to use for the extra samples
                (n_workers <= 1 runs them sequentially with the shared global
                random state, identical to the pre-parallelism behavior)
            rollout_aggregation: 'max' (default) applies the
                rollout_discount**rollout_depth discount to the extra samples;
                'mean' leaves them undiscounted
            rollout_discount: Base of the depth discount applied under 'max'
                aggregation (default: 0.9). Set to 1.0 to disable the discount
                entirely - extra samples are then compared to the depth=0
                sample on equal footing before taking the max. Unused under
                'mean' aggregation.

        Returns:
            List of reward values, one per rollout sample
        """
        rewards = [new_node.rollout(depth=0, energy_calculator=energy_calculator,
                                     mode=mode, doscar_lookup=doscar_lookup)]

        n_extra = n_rollout - 1
        if n_extra <= 0:
            return rewards

        scale = rollout_discount ** rollout_depth if rollout_aggregation == 'max' else 1.0

        if n_workers <= 1:
            for _ in range(n_extra):
                rollout_reward = new_node.rollout(
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode=mode,
                    doscar_lookup=doscar_lookup
                )
                rewards.append(scale * rollout_reward)
            return rewards

        # Draw per-task seeds up front (sequentially, from the shared global
        # random state) so the set of substitutions tried is deterministic
        # under --seed no matter how the thread pool schedules the tasks.
        task_seeds = [random.randint(0, 2**31 - 1) for _ in range(n_extra)]

        with ThreadPoolExecutor(max_workers=min(n_workers, n_extra)) as executor:
            futures = [
                executor.submit(
                    new_node.rollout,
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode=mode,
                    doscar_lookup=doscar_lookup,
                    rng=random.Random(seed)
                )
                for seed in task_seeds
            ]
            rewards.extend(scale * future.result() for future in futures)

        return rewards

    def expansion_simulation(self, rollout_depth: int = 1, n_rollout: int = 1,
                           energy_calculator=None, rollout_method: str = 'ehull',
                           beta: float = 1.0, gamma: float = 0.0001,
                           doscar_lookup=None, n_workers: int = 1,
                           rollout_aggregation: str = 'max',
                           rollout_discount: float = 0.9) -> Tuple[float, bool]:
        """
        Expand selected node and perform rollout simulation.

        Args:
            rollout_depth: Depth of rollout simulation
            n_rollout: Number of rollout simulations
            energy_calculator: Energy calculator instance
            rollout_method: Rollout evaluation method ('ehull', 'ehull_rdos', or 'rdos')
            beta: Weight for E_hull reward when using 'ehull_rdos' method (default: 1.0)
            gamma: Weight for rDOS reward when using 'ehull_rdos' method (default: 0.0001)
            doscar_lookup: DoscarRewardLookup instance for DOSCAR rewards
            n_workers: Number of worker threads for the n_rollout samples
                (default: 1, i.e. sequential, identical to prior behavior)
            rollout_aggregation: How to combine the n_rollout reward samples into
                this node's reward - 'max' (default, original behavior: optimistic,
                biased upward by however many samples are drawn, with the extra
                samples discounted by rollout_discount**rollout_depth before
                comparison) or 'mean' (plain average of undiscounted samples)
            rollout_discount: Base of the depth discount under 'max' aggregation
                (default: 0.9). Set to 1.0 to disable discount. See
                _run_rollout_samples() for details.

        Returns:
            Tuple of (reward, renew_t_to_terminate_flag)
        """
        renew_t_to_terminate = False
        
        # Expand node if not already expanded
        if not self.current_node.children:
            if not self.current_node.expansion_list:
                self.current_node.expand()
            
        # Prefer candidates whose formula has not yet been seen anywhere in the
        # tree (in stat_dict). This prevents wasting an expansion slot on a
        # composition the search has already evaluated via a different parent,
        # improving coverage of genuinely new chemical space. If every candidate
        # in the expansion list has already been seen (the node is in a
        # densely-explored neighbourhood), fall back to the full list so the
        # search isn't stuck - the selected node inherits the termination status
        # already recorded in stat_dict.
        unseen = [n for n in self.current_node.expansion_list
                  if n.get_chemical_formula() not in self.stat_dict]
        draw_pool = unseen if unseen else self.current_node.expansion_list
        new_node = random.choice(draw_pool)
        if not unseen:
            # All candidates already seen - inherit recorded termination status
            formula = new_node.get_chemical_formula()
            if formula in self.stat_dict:
                new_node.terminated = self.stat_dict[formula][2]

        # Remove from expansion list and add to tree
        self.current_node.expansion_list.remove(new_node)
        new_node.add_parent(self.current_node)
        self.current_node.add_child(new_node)
        self.current_node.update_expandable()
        
        # Perform rollout simulations
        if rollout_method == 'ehull':
            # E_hull only (tanh-transformed), no DFT/DOSCAR data required
            mode = 'ehull'
        elif rollout_method == 'ehull_rdos':
            # E_hull (tanh-transformed) + rDOS, requires doscar_peaks_data_with_U.csv
            mode = f'ehull_rdos_{beta}_{gamma}'
        elif rollout_method == 'rdos':
            # rDOS only, requires doscar_peaks_data_with_U.csv, no MACE/Materials Project needed
            mode = 'rdos'
        else:
            raise ValueError(f"Unknown rollout_method: {rollout_method}")

        rewards = self._run_rollout_samples(
            new_node, rollout_depth, n_rollout, energy_calculator, mode,
            doscar_lookup, n_workers, rollout_aggregation, rollout_discount
        )

        if rollout_aggregation == 'max':
            reward = np.max(rewards)
        elif rollout_aggregation == 'mean':
            reward = np.mean(rewards)
        else:
            raise ValueError(f"Unknown rollout_aggregation: {rollout_aggregation}")
        extra = 0
        
        # Check for new maximum reward
        if reward >= self.max_reward:
            if self.max_reward > 0:
                extra = 1
            self.max_reward = reward
            renew_t_to_terminate = True
            self.best_node = new_node
            
        reward += extra
        new_node.update_rewards(reward)
        new_node.visit(renew_t_to_terminate)
        self.current_node = new_node
        
        return reward, renew_t_to_terminate
        
    def stat_node_visited(self):
        """
        Record statistics for visited nodes.
        """
        formula = self.current_node.get_chemical_formula()

        if formula not in self.stat_dict:
            # Get DOS reward if available
            dos_reward = 0.0
            if hasattr(self, 'doscar_lookup') and self.doscar_lookup is not None:
                dos_reward = self.doscar_lookup.get_reward(formula)

            self.stat_dict[formula] = [
                self.current_node.get_rewards(total=False),
                0,
                False,
                self.current_node.e_above_hull,
                self.current_node.e_form,
                dos_reward
            ]

        self.stat_dict[formula][1] += 1
        
    def _probability_selector(self, ucb_values: List[float]) -> int:
        """
        Select index based on probability proportional to UCB values.
        
        Args:
            ucb_values: List of UCB values
            
        Returns:
            Selected index
        """
        ucb_processed = []
        for value in ucb_values:
            if value > 0:
                ucb_processed.append(value)
            else:
                ucb_processed.append(np.exp(value))
                
        weights = np.cumsum(np.square(ucb_processed))
        random_value = random.random() * weights[-1]

        for i, weight in enumerate(weights):
            if weight > random_value:
                return i
        return len(weights) - 1

    def _uniform_selector(self, scores: List[float]) -> int:
        """
        Select an index uniformly at random, ignoring score magnitude entirely
        (the exploratory step of textbook epsilon-greedy). Terminated children
        (sentinel score -1e4) are excluded so exploration doesn't waste rollouts
        on dead branches.

        Args:
            scores: List of per-child scores (UCB1 values, with -1e4 for
                terminated children)

        Returns:
            Selected index
        """
        candidates = [i for i, s in enumerate(scores) if s != -1e4]
        if not candidates:
            candidates = list(range(len(scores)))
        return random.choice(candidates)

    def _boltzmann_selector(self, scores: List[float]) -> int:
        """
        Select an index with probability proportional to exp(score / temperature)
        (softmax/Boltzmann exploration). Lower temperature biases toward greedy
        argmax; higher temperature biases toward uniform random.

        Args:
            scores: List of per-child UCB1 values (with -1e4 for terminated
                children, and possibly +inf for never-visited children)

        Returns:
            Selected index
        """
        arr = np.array(scores, dtype=float)

        # Never-visited children have UCB1 = +inf; always explore one of
        # those before weighing finite values via softmax.
        inf_mask = np.isinf(arr) & (arr > 0)
        if inf_mask.any():
            candidates = np.flatnonzero(inf_mask)
            return int(np.random.choice(candidates))

        scaled = arr / self.temperature
        scaled -= scaled.max()  # numerical stability; doesn't change the softmax result
        weights = np.exp(scaled)
        probs = weights / weights.sum()
        return int(np.random.choice(len(arr), p=probs))


    def run(self, n_iterations: int, energy_calculator=None,
            rollout_depth: int = 1, n_rollout: int = 10,
            selection_mode: str = 'ucb1', rollout_method: str = 'ehull',
            beta: float = 1.0, gamma: float = 0.0001,
            doscar_lookup=None, n_workers: int = 1,
            rollout_aggregation: str = 'max',
            rollout_discount: float = 0.9) -> Dict:
        """
        Run MCTS algorithm for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations
            energy_calculator: Energy calculator instance
            rollout_depth: Depth of rollout simulations
            n_rollout: Number of rollout simulations per expansion
            selection_mode: Child-selection strategy for the selection phase.
                One of 'ucb1' (default), 'epsilon_greedy', 'boltzmann', 'puct',
                or 'hybrid'. See select_node() for details on each.
            rollout_method: Rollout evaluation method ('ehull', 'ehull_rdos', or 'rdos')
            beta: Weight for E_hull reward when using 'ehull_rdos' method (default: 1.0)
            gamma: Weight for rDOS reward when using 'ehull_rdos' method (default: 0.0001)
            doscar_lookup: DoscarRewardLookup instance for DOSCAR rewards
            rollout_aggregation: How to combine a node's n_rollout reward samples -
                'max' (default) or 'mean'. See expansion_simulation() for details.
                  'ehull':      reward = ehull_reward(e_above_hull)
                  'ehull_rdos': reward = beta*ehull_reward(e_above_hull) + gamma*r_DOS
                  'rdos':       reward = r_DOS
            rollout_discount: Base of the depth discount under 'max' aggregation
                (default: 0.9). Set to 1.0 to disable the discount entirely.
                Unused under 'mean' aggregation.
            n_workers: Number of worker threads to evaluate each expansion's
                n_rollout samples concurrently (default: 1, i.e. sequential,
                identical to prior behavior). For a given n_workers value,
                repeated runs with the same --seed reproduce identical results
                (each parallel sample draws from its own deterministically-
                seeded RNG, independent of thread scheduling order). Note
                results are NOT expected to match across different n_workers
                values for the same seed, since the parallel and sequential
                code paths consume the shared global RNG differently.

        Returns:
            Dictionary containing run statistics
        """
        # Store doscar_lookup for statistics collection
        self.doscar_lookup = doscar_lookup

        # Initialize tracking with root node (iteration 0)
        self.best_e_form_node = self.root
        self.best_e_hull_node = self.root
        self.best_e_form_history.append(self.root.e_form)
        self.best_e_form_e_hull_history.append(self.root.e_above_hull)
        self.best_e_form_formula_history.append(self.root.get_chemical_formula())
        self.best_e_hull_history.append(self.root.e_above_hull)
        self.best_e_hull_e_form_history.append(self.root.e_form)
        self.best_e_hull_formula_history.append(self.root.get_chemical_formula())

        # Initialize rDOS tracking with root node
        root_rdos = 0.0
        if doscar_lookup is not None:
            root_rdos = doscar_lookup.get_reward(self.root.get_chemical_formula())
        self.best_rdos_node = self.root
        self.best_rdos_value = root_rdos
        self.best_rdos_history.append(root_rdos)
        self.best_rdos_eform_history.append(self.root.e_form)
        self.best_rdos_ehull_history.append(self.root.e_above_hull)
        self.best_rdos_formula_history.append(self.root.get_chemical_formula())

        self.n_unique_compounds_history.append(1)  # Start with root node only
        self.max_reward_history.append(self.max_reward)

        for i in range(n_iterations):
            if self.terminated:
                break

            # Selection
            select_chain = self.select_node(mode=selection_mode)

            if self.terminated:
                break

            # Record statistics
            self.stat_node_visited()

            # Expansion and simulation
            reward, renew_t_to_terminate = self.expansion_simulation(
                rollout_depth=rollout_depth,
                n_rollout=n_rollout,
                energy_calculator=energy_calculator,
                rollout_method=rollout_method,
                beta=beta,
                gamma=gamma,
                doscar_lookup=doscar_lookup,
                n_workers=n_workers,
                rollout_aggregation=rollout_aggregation,
                rollout_discount=rollout_discount
            )

            # Back-propagation
            self.back_propagation(reward, select_chain, renew_t_to_terminate)

            # Update statistics
            self.stat_node_visited()

            # Track node with best (minimum) E_form
            if self.best_e_form_node is None:
                self.best_e_form_node = self.current_node
            elif self.current_node.e_form < self.best_e_form_node.e_form:
                self.best_e_form_node = self.current_node

            # Track node with best (minimum) E_hull
            if self.best_e_hull_node is None:
                self.best_e_hull_node = self.current_node
            elif self.current_node.e_above_hull < self.best_e_hull_node.e_above_hull:
                self.best_e_hull_node = self.current_node

            # Track node with best (maximum) rDOS
            current_rdos = 0.0
            if doscar_lookup is not None:
                current_rdos = doscar_lookup.get_reward(self.current_node.get_chemical_formula())
            if self.best_rdos_node is None:
                self.best_rdos_node = self.current_node
                self.best_rdos_value = current_rdos
            elif current_rdos > self.best_rdos_value:
                self.best_rdos_node = self.current_node
                self.best_rdos_value = current_rdos

            # Record convergence history for both compounds
            self.best_e_form_history.append(self.best_e_form_node.e_form)
            self.best_e_form_e_hull_history.append(self.best_e_form_node.e_above_hull)
            self.best_e_form_formula_history.append(self.best_e_form_node.get_chemical_formula())

            self.best_e_hull_history.append(self.best_e_hull_node.e_above_hull)
            self.best_e_hull_e_form_history.append(self.best_e_hull_node.e_form)
            self.best_e_hull_formula_history.append(self.best_e_hull_node.get_chemical_formula())

            # Record rDOS convergence history
            self.best_rdos_history.append(self.best_rdos_value)
            self.best_rdos_eform_history.append(self.best_rdos_node.e_form)
            self.best_rdos_ehull_history.append(self.best_rdos_node.e_above_hull)
            self.best_rdos_formula_history.append(self.best_rdos_node.get_chemical_formula())

            # Track number of unique compounds discovered
            self.n_unique_compounds_history.append(len(self.stat_dict))

            # Track best composite reward (the actual MCTS objective) so far
            self.max_reward_history.append(self.max_reward)

        # Final statistics
        results = {
            'iterations_completed': i + 1 if not self.terminated else i,
            'best_reward': self.max_reward,
            'best_node_formula': self.best_node.get_chemical_formula() if self.best_node else None,
            'best_node_e_form': self.best_node.e_form if self.best_node else None,
            'best_node_e_above_hull': self.best_node.e_above_hull if self.best_node else None,
            'stat_dict': self.stat_dict.copy(),
            'terminated': self.terminated
        }
        
        return results
        
    def get_statistics_dataframe(self) -> pd.DataFrame:
        """
        Convert statistics dictionary to DataFrame.

        Returns:
            DataFrame with statistics
        """
        stat_df = stat_dict_to_dataframe(self.stat_dict)
        if not stat_df.empty:
            stat_df = stat_df.sort_values(by='visit_count', ascending=False)
        return stat_df
        
    def save_statistics(self, filename: str):
        """
        Save statistics to CSV file.

        Args:
            filename: Output filename
        """
        stat_df = self.get_statistics_dataframe()
        stat_df.to_csv(filename)

    def save_convergence_history(self, filename: str):
        """
        Save convergence history (best composite reward, E_form, E_hull, and
        rDOS per iteration) to CSV file. 'best_reward' is max_reward - the best
        composite reward (the actual MCTS objective) found so far at each
        iteration. Also tracks three separate compounds at each iteration:
        1. Compound with minimum E_form (and its E_hull)
        2. Compound with minimum E_hull (and its E_form)
        3. Compound with maximum rDOS (and its E_form, E_hull)

        Args:
            filename: Output filename
        """
        convergence_df = pd.DataFrame({
            'iteration': list(range(len(self.best_e_form_history))),
            'n_unique_compounds': self.n_unique_compounds_history,
            'best_reward': self.max_reward_history,
            'best_e_form': self.best_e_form_history,
            'best_e_form_e_hull': self.best_e_form_e_hull_history,
            'best_e_form_formula': self.best_e_form_formula_history,
            'best_e_hull': self.best_e_hull_history,
            'best_e_hull_e_form': self.best_e_hull_e_form_history,
            'best_e_hull_formula': self.best_e_hull_formula_history,
            'best_rdos': self.best_rdos_history,
            'best_rdos_eform': self.best_rdos_eform_history,
            'best_rdos_ehull': self.best_rdos_ehull_history,
            'best_rdos_formula': self.best_rdos_formula_history
        })

        # Post-process: Update energies from stat_dict (which has the actual calculated values)
        convergence_df = self._postprocess_convergence_energies(convergence_df)

        convergence_df.to_csv(filename, index=False)
        logger.info(f"Saved convergence history to {filename}")

    def _postprocess_convergence_energies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process convergence history to ensure all formulas have correct energies.
        Looks up actual E_form and E_hull values from stat_dict.

        Args:
            df: Convergence history DataFrame

        Returns:
            Updated DataFrame with correct energies
        """
        if not self.stat_dict:
            return df

        # Create a copy to avoid modifying the original
        df = df.copy()

        # For each unique formula in the convergence history, look up its actual energies
        for idx, row in df.iterrows():
            # Update best_e_form compound energies
            formula_eform = row['best_e_form_formula']
            if formula_eform in self.stat_dict:
                df.at[idx, 'best_e_form'] = self.stat_dict[formula_eform][4]  # e_form is at index 4
                df.at[idx, 'best_e_form_e_hull'] = self.stat_dict[formula_eform][3]  # e_above_hull is at index 3

            # Update best_e_hull compound energies
            formula_ehull = row['best_e_hull_formula']
            if formula_ehull in self.stat_dict:
                df.at[idx, 'best_e_hull'] = self.stat_dict[formula_ehull][3]  # e_above_hull is at index 3
                df.at[idx, 'best_e_hull_e_form'] = self.stat_dict[formula_ehull][4]  # e_form is at index 4

            # Update best_rdos compound energies
            formula_rdos = row['best_rdos_formula']
            if formula_rdos in self.stat_dict:
                df.at[idx, 'best_rdos_eform'] = self.stat_dict[formula_rdos][4]  # e_form is at index 4
                df.at[idx, 'best_rdos_ehull'] = self.stat_dict[formula_rdos][3]  # e_above_hull is at index 3
                # Update rDOS value if available (index 5 when 6 elements exist)
                if len(self.stat_dict[formula_rdos]) >= 6:
                    df.at[idx, 'best_rdos'] = self.stat_dict[formula_rdos][5]

        return df