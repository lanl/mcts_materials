"""
MCTS Tree Node implementation for crystal structure optimization.
"""

import numpy as np
from typing import List, Optional, Tuple
from ase import Atoms


def ehull_reward(e_hull: float) -> float:
    """
    Sharp tanh-based reward for energy above hull.

    Uses f(E_hull) = -tanh(120 * (E_hull - 0.05))

    This gives a sharp transition around 0.05 eV/atom:
    - At E_hull = 0.05: reward = 0 (boundary)
    - At E_hull = 0 (stable): reward ≈ +1 (favorable)
    - At E_hull = 0.1 (unstable): reward ≈ -1 (unfavorable)

    Args:
        e_hull: Energy above hull value (eV/atom)

    Returns:
        Reward value in range [-1, +1]
    """
    return -np.tanh(120.0 * (e_hull - 0.05))


class MCTSTreeNode:
    """
    MCTS tree node representing a crystal structure.
    
    Each node contains an ASE Atoms object and manages its relationships
    in the MCTS tree structure.
    """
    
    def __init__(self, atoms: Atoms, f_block_mode: str = 'u_only', exploration_constant: float = 0.1, termination_limit: int = 60, move_step: int = 1):
        """
        Initialize an MCTS tree node.

        Args:
            atoms: ASE Atoms object representing the crystal structure
            f_block_mode: F-block substitution mode ('u_only', 'full_f_block', 'experimental',
                'lanthanides_u', or 'lanthanides_u_extended')
            exploration_constant: Exploration constant for UCB calculation (default: 0.1)
            termination_limit: Number of visits before terminating a node (default: 60)
        """
        self.atoms = atoms
        self.symbols = atoms.symbols
        self.f_block_mode = f_block_mode
        self.exploration_constant = exploration_constant
        self.termination_limit = termination_limit
        self.move_step = move_step
        self.parent: Optional['MCTSTreeNode'] = None
        self.children: List['MCTSTreeNode'] = []
        self.expandable = True
        self.g_iv = 0
        self.metal = 0
        self.f_block = 0
        self.e_form = 0.0
        self.e_above_hull = 0.0
        self.expansion_list: List['MCTSTreeNode'] = []

        # MCTS statistics
        self.t_of_visit = 0
        self.total_reward = 0.0
        self.best_reward = -10.0
        self.terminated = False
        self.t_to_terminate = termination_limit
        
        # Initialize possible moves
        self._determine_possible_moves()
        # Don't auto-expand to avoid recursion issues
        # expand() will be called explicitly when needed
        
    def substitute(self, metal: int, g_iv: int, f_block: Optional[int] = None) -> Atoms:
        """
        Substitute current atoms with other transition metal, group IV elements, and f-block elements.
        
        Args:
            metal: Atomic number of the transition metal
            g_iv: Atomic number of the group IV element
            f_block: Atomic number of the f-block element (optional)
            
        Returns:
            New ASE Atoms object with substitutions
        """
        op_mat = []
        g_iv_list = [14, 32, 50, 82]  # Si, Ge, Sn, Pb
        f_block_list = (list(range(57, 72)) + list(range(89, 95)))  # Lanthanides + allowed actinides
        
        for atomic_num in self.atoms.get_atomic_numbers():
            if atomic_num in f_block_list and f_block is not None:
                # Substitute f-block elements
                op_mat.append(f_block - atomic_num)
            elif atomic_num in g_iv_list:
                # Substitute Group IV elements  
                op_mat.append(g_iv - atomic_num)
            else:
                # Substitute transition metals
                op_mat.append(metal - atomic_num)
        
        new_atoms = self.atoms.copy()
        new_atoms.set_atomic_numbers(new_atoms.get_atomic_numbers() + op_mat)
        return new_atoms
        
    def _determine_possible_moves(self):
        """
        Determine possible moves for transition metals, group IV elements, and f-block elements.

        For transition metals: can move up, down, left, right on periodic table
        For group IV elements: adjacent chain only (Si <-> Ge <-> Sn <-> Pb)
        For f-block elements: extended moves (±3) for better exploration
        """
        g_iv_chain = [14, 32, 50, 82]  # Si, Ge, Sn, Pb, in chain order
        self.g_iv_move = g_iv_chain
        self.f_block_move = []  # Will be set based on current f-block element

        for atomic_num in set(self.atoms.get_atomic_numbers()):
            if atomic_num in g_iv_chain:
                self.g_iv = atomic_num
                # Chain-restricted: only the current element and its immediate
                # neighbors in the Si-Ge-Sn-Pb chain (no direct jumps, e.g.
                # Sn->Si requires passing through Ge first)
                idx = g_iv_chain.index(atomic_num)
                moves = [atomic_num]
                for delta in range(1, self.move_step + 1):
                    if idx - delta >= 0:
                        moves.append(g_iv_chain[idx - delta])
                    if idx + delta < len(g_iv_chain):
                        moves.append(g_iv_chain[idx + delta])
                self.g_iv_move = sorted(moves)
            elif 22 <= atomic_num <= 30:  # 3d transition metals
                self.metal = atomic_num
                mv = {atomic_num}
                for s in range(1, self.move_step + 1):
                    if atomic_num - s >= 22:
                        mv.add(atomic_num - s)
                    if atomic_num + s <= 30:
                        mv.add(atomic_num + s)
                mv.add(atomic_num + 18)  # cross-period to 4d (always single period step)
                self.metal_move = sorted(mv)
            elif 40 <= atomic_num <= 48:  # 4d transition metals
                self.metal = atomic_num
                mv = {atomic_num}
                for s in range(1, self.move_step + 1):
                    if atomic_num - s >= 40:
                        mv.add(atomic_num - s)
                    if atomic_num + s <= 48:
                        mv.add(atomic_num + s)
                mv.add(atomic_num - 18)  # cross-period down to 3d
                if atomic_num + 32 <= 80:
                    mv.add(atomic_num + 32)  # cross-period up to 5d
                self.metal_move = sorted(mv)
            elif 72 <= atomic_num <= 80:  # 5d transition metals
                self.metal = atomic_num
                mv = {atomic_num}
                for s in range(1, self.move_step + 1):
                    if atomic_num - s >= 72:
                        mv.add(atomic_num - s)
                    if atomic_num + s <= 80:
                        mv.add(atomic_num + s)
                mv.add(atomic_num - 32)  # cross-period down to 4d (always single period step)
                self.metal_move = sorted(mv)
            elif 57 <= atomic_num <= 71 or 89 <= atomic_num <= 94:  # f-block elements (lanthanides + actinides, excluding 95-103)
                self.f_block = atomic_num
                self._determine_f_block_moves(atomic_num)
                    
    def _determine_f_block_moves(self, atomic_num: int):
        """
        Determine possible f-block element moves based on the f_block_mode.

        Args:
            atomic_num: Current f-block atomic number
        """
        if self.f_block_mode == 'u_only':
            # U-only mode: restrict moves to only U (92)
            possible_moves = [92]  # Only U allowed
        elif self.f_block_mode == 'lanthanides_u_extended':
            # Extended Lanthanides + U mode: uses self.move_step for lanthanide jumps.
            lanthanides = list(range(58, 72))  # Ce (58) to Lu (71)

            possible_moves = [atomic_num]

            if atomic_num in lanthanides:
                idx = lanthanides.index(atomic_num)
                for delta in range(-self.move_step, self.move_step + 1):
                    if delta == 0:
                        continue
                    neighbor_idx = (idx + delta) % len(lanthanides)
                    possible_moves.append(lanthanides[neighbor_idx])

            # Allow transitions between lanthanides and U
            if atomic_num == 92:
                possible_moves.extend([60, 64, 68])  # Nd, Gd, Er (light, mid, heavy)
            elif atomic_num in [60, 64, 68]:
                possible_moves.append(92)
        elif self.f_block_mode == 'lanthanides_u':
            # Lanthanides + U mode: uses self.move_step for lanthanide jumps.
            lanthanides = list(range(58, 72))  # Ce (58) to Lu (71)

            possible_moves = [atomic_num]

            if atomic_num in lanthanides:
                idx = lanthanides.index(atomic_num)
                for delta in range(-self.move_step, self.move_step + 1):
                    if delta == 0:
                        continue
                    possible_moves.append(lanthanides[(idx + delta) % len(lanthanides)])

            # Allow transitions between lanthanides and U
            if atomic_num == 92:
                possible_moves.append(60)
            elif atomic_num == 60:
                possible_moves.append(92)
        elif self.f_block_mode == 'experimental':
            # Experimental mode: actinides (minus La) plus U, allowing adjacent comparisons
            lanthanides_no_la = list(range(58, 72))  # Ce (58) to Lu (71), excludes La (57)
            
            # Start with the current element
            possible_moves = [atomic_num]
            
            # Add adjacent elements (±1) if they exist and are in our allowed set
            for delta in [-1, +1]:
                neighbor = atomic_num + delta
                if neighbor in lanthanides_no_la:
                    possible_moves.append(neighbor)
            # Add U (92) if not already included 
            if atomic_num == 92:
                possible_moves.append(60)
            elif atomic_num == 60:
                possible_moves.append(92)
        else:
            # Full f-block mode: original implementation with lanthanides + allowed actinides
            lanthanides = list(range(58, 72))  # Ce (58) to Lu (71)
            actinides = list(range(90, 95))    # Th (90) to Pu (94)
            all_f_elements = lanthanides + actinides
            
            # Start with the current element
            possible_moves = [atomic_num]
            
            # Add adjacent elements (±1) if they exist and are allowed
            for delta in [-1, +1]:
                neighbor = atomic_num + delta
                if neighbor in all_f_elements:
                    possible_moves.append(neighbor)
            
            # For lanthanides, allow "vertical" moves to corresponding actinides
            # Ce(58)->Th(90), Pr(59)->Pa(91), Nd(60)->U(92), Pm(61)->Np(93), Sm(62)->Pu(94)
            if 58 <= atomic_num <= 62:  # Ce to Sm
                actinide_analog = atomic_num + 32  # Ce(58)+32=Th(90), etc.
                if actinide_analog in all_f_elements:
                    possible_moves.append(actinide_analog)
                    
            # For actinides, allow "vertical" moves to corresponding lanthanides  
            if 90 <= atomic_num <= 94:  # Th to Pu
                lanthanide_analog = atomic_num - 32  # Th(90)-32=Ce(58), etc.
                if lanthanide_analog in all_f_elements:
                    possible_moves.append(lanthanide_analog)

        self.f_block_move = sorted(list(set(possible_moves)))
    
    def expand(self):
        """
        Create the expansion list for this node by generating all possible substitutions.
        """
        expansion_list = []
        
        # Generate substitutions including f-block elements
        f_block_options = getattr(self, 'f_block_move', [None])
        if not f_block_options:
            f_block_options = [None]
            
        for metal in self.metal_move:
            for g_iv in self.g_iv_move:
                for f_block in f_block_options:
                    new_atoms = self.substitute(metal, g_iv, f_block)
                    # Create new node without auto-expanding to avoid infinite recursion
                    new_node = MCTSTreeNode.__new__(MCTSTreeNode)
                    new_node.atoms = new_atoms
                    new_node.symbols = new_atoms.symbols
                    new_node.f_block_mode = self.f_block_mode
                    new_node.exploration_constant = self.exploration_constant
                    new_node.termination_limit = self.termination_limit
                    new_node.move_step = self.move_step
                    new_node.parent = None
                    new_node.children = []
                    new_node.expandable = True
                    new_node.g_iv = 0
                    new_node.metal = 0
                    new_node.f_block = 0
                    new_node.e_form = 0.0
                    new_node.e_above_hull = 0.0
                    new_node.expansion_list = []
                    new_node.t_of_visit = 0
                    new_node.total_reward = 0.0
                    new_node.best_reward = -10.0
                    new_node.terminated = False
                    new_node.t_to_terminate = self.termination_limit
                    new_node._determine_possible_moves()
                    expansion_list.append(new_node)
        
        # Remove identical compositions to parent (avoid going backwards)
        if self.parent is not None:
            expansion_list = [
                node for node in expansion_list 
                if node.symbols.get_chemical_formula() != self.parent.symbols.get_chemical_formula()
            ]
        
        self.expansion_list = expansion_list
        
    def rollout(self, depth: int = 1, energy_calculator=None, mode: str = 'ehull', doscar_lookup=None, rng=None) -> float:
        """
        Perform rollout simulation from this node.

        depth=0: evaluate the node's own composition and record e_form/e_above_hull as
            a side effect. This is always the first sample in _run_rollout_samples.

        depth>0: max-along-walk. Take `depth` independent random substitution steps,
            evaluate the reward at *each* intermediate composition, and return the
            maximum reward seen across all steps. Every composition along the walk is
            a valid candidate, so scoring only the endpoint discards information;
            max-along-walk extracts `depth` candidate evaluations per walk instead of 1.

        Args:
            depth: Number of random substitution steps. 0 evaluates the node itself;
                >0 performs a random walk and returns the max reward along it.
            energy_calculator: Energy calculator instance (required for 'ehull'/'ehull_rdos',
                unused for 'rdos')
            mode: Evaluation mode. One of:
                - 'ehull': reward = ehull_reward(e_above_hull)
                - 'ehull_rdos_{beta}_{gamma}': reward = beta*ehull_reward(e_above_hull) + gamma*r_DOS
                - 'rdos': reward = r_DOS only
            doscar_lookup: DoscarRewardLookup instance for DOSCAR-derived rDOS rewards
            rng: Random source exposing .choice(). Defaults to the shared `random` module.

        Returns:
            Reward value (scalar)
        """
        import random
        rng = rng if rng is not None else random

        tmp_atoms = self.atoms.copy()
        tmp_metal_move = self.metal_move.copy()
        tmp_g_iv_move = self.g_iv_move.copy()
        tmp_f_block_move = getattr(self, 'f_block_move', [None])

        def _score(atoms, record=False):
            """Compute the reward for `atoms` under the current mode.
            When record=True, also stores e_form/e_above_hull on this node (depth=0 only)."""
            if mode == 'rdos':
                if doscar_lookup is not None:
                    return doscar_lookup.get_reward(atoms.get_chemical_formula(mode='metal'))
                return 0.0
            if energy_calculator is None:
                return 0.0
            e_form, e_above_hull = energy_calculator.calculate_energies(atoms)
            if record:
                # e_form is tracked for reference/reporting only; not part of the reward
                self.e_form = e_form
                self.e_above_hull = e_above_hull
            if mode == 'ehull':
                return ehull_reward(e_above_hull)
            elif mode.startswith('ehull_rdos'):
                try:
                    parts = mode.split('_')
                    beta = float(parts[2])
                    gamma = float(parts[3]) if len(parts) > 3 else 0.0
                except (IndexError, ValueError):
                    beta = 1.0
                    gamma = 0.0001
                doscar_reward = 0.0
                if gamma > 0 and doscar_lookup is not None:
                    doscar_reward = doscar_lookup.get_reward(atoms.get_chemical_formula(mode='metal'))
                return beta * ehull_reward(e_above_hull) + gamma * doscar_reward
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # depth=0: evaluate the node itself (side effect: records e_form/e_above_hull)
        if depth == 0:
            return _score(tmp_atoms, record=True)

        # depth>0: max-along-walk — evaluate at every step, return the maximum
        g_iv_list = [14, 32, 50, 82]
        f_block_list = list(range(57, 72)) + list(range(89, 95))
        step_rewards = []
        for _ in range(depth):
            metal = rng.choice(tmp_metal_move)
            g_iv = rng.choice(tmp_g_iv_move)
            f_block = rng.choice(tmp_f_block_move) if tmp_f_block_move != [None] else None

            op_mat = []
            for atomic_num in tmp_atoms.get_atomic_numbers():
                if atomic_num in f_block_list and f_block is not None:
                    op_mat.append(f_block - atomic_num)
                elif atomic_num in g_iv_list:
                    op_mat.append(g_iv - atomic_num)
                else:
                    op_mat.append(metal - atomic_num)

            tmp_atoms = tmp_atoms.copy()
            tmp_atoms.set_atomic_numbers(tmp_atoms.get_atomic_numbers() + op_mat)
            step_rewards.append(_score(tmp_atoms, record=False))

        return max(step_rewards)
            
    def update_rewards(self, reward: float):
        """Update the rewards for this node."""
        self.total_reward += reward
        if reward > self.best_reward:
            self.best_reward = reward
            
    def get_rewards(self, total: bool = True) -> float:
        """Get rewards for this node."""
        return self.total_reward if total else self.best_reward
        
    def get_ucb(self) -> float:
        """
        Calculate Upper Confidence Bound (UCB) value for this node.
        
        Returns:
            UCB value
        """
        if self.t_of_visit == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.t_of_visit
        exploration = self.exploration_constant * np.sqrt(np.log(self.parent.t_of_visit) / self.t_of_visit)

        return exploitation + exploration

    def get_puct(self, prior: float) -> float:
        """
        Calculate the PUCT (Predictor + UCB applied to Trees) value for this node,
        AlphaZero-style. Unlike get_ucb(), an unvisited node has Q=0 rather than
        +inf - exploration is driven entirely by the (1 + N) term in the
        denominator, which is largest when N=0.

        Since this codebase has no learned policy network, `prior` is a uniform
        prior (1 / number of siblings) rather than a predicted move probability.

        Args:
            prior: Prior probability assigned to this node (uniform, since there
                is no policy network to predict one)

        Returns:
            PUCT value
        """
        q = self.total_reward / self.t_of_visit if self.t_of_visit > 0 else 0.0
        exploration = self.exploration_constant * prior * np.sqrt(self.parent.t_of_visit) / (1 + self.t_of_visit)

        return q + exploration

    def visit(self, renew_t_to_terminate: bool = False):
        """
        Update visit count and termination countdown.
        
        Args:
            renew_t_to_terminate: Whether to reset termination countdown
        """
        self.t_of_visit += 1

        if renew_t_to_terminate:
            self.t_to_terminate = self.termination_limit
        else:
            self.t_to_terminate -= 1
            
        self._check_termination()
        
    def _check_termination(self):
        """Check if node should be terminated."""
        if self.t_to_terminate <= 0:
            self.terminated = True
            
    def add_parent(self, parent: 'MCTSTreeNode'):
        """Add parent node."""
        self.parent = parent
        
    def add_child(self, child: 'MCTSTreeNode'):
        """Add child node."""
        self.children.append(child)
        
    def update_expandable(self):
        """Update expandable status based on remaining expansion candidates."""
        self.expandable = len(self.expansion_list) > 0
        
    def get_chemical_formula(self) -> str:
        """Get chemical formula of this node."""
        return self.atoms.get_chemical_formula()