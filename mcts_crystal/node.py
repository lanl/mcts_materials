"""
MCTS Tree Node implementation for crystal structure optimization.
"""

import numpy as np
from typing import List, Optional, Tuple
from ase import Atoms


class MCTSTreeNode:
    """
    MCTS tree node representing a crystal structure.
    
    Each node contains an ASE Atoms object and manages its relationships
    in the MCTS tree structure.
    """
    
    def __init__(self, atoms: Atoms, f_block_mode: str = 'u_only', exploration_constant: float = 0.1):
        """
        Initialize an MCTS tree node.
        
        Args:
            atoms: ASE Atoms object representing the crystal structure
            f_block_mode: F-block substitution mode ('u_only', 'full_f_block', or 'experimental')
            exploration_constant: Exploration constant for UCB calculation (default: 0.1)
        """
        self.atoms = atoms
        self.symbols = atoms.symbols
        self.f_block_mode = f_block_mode
        self.exploration_constant = exploration_constant
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
        self.t_to_terminate = 30
        
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
        For group IV elements: can move up and down within the group
        For f-block elements: adjacent moves excluding elements 95-103
        """
        self.g_iv_move = [14, 32, 50, 82]  # Si, Ge, Sn, Pb
        self.f_block_move = []  # Will be set based on current f-block element
        
        for atomic_num in set(self.atoms.get_atomic_numbers()):
            if atomic_num in self.g_iv_move:
                self.g_iv = atomic_num
                # Restrict moves based on current position
                if atomic_num == 14:  # Si
                    self.g_iv_move = [14, 32]
                elif atomic_num == 32:  # Ge
                    self.g_iv_move = [14, 32, 50]
                elif atomic_num == 50:  # Sn
                    self.g_iv_move = [32, 50, 82]
                elif atomic_num == 82:  # Pb
                    self.g_iv_move = [50, 82]
            elif 22 <= atomic_num <= 30:  # 3d transition metals
                self.metal = atomic_num
                if atomic_num == 22:  # Ti
                    self.metal_move = [22, 23, 40]  # Ti, V, Zr
                elif atomic_num == 30:  # Zn
                    self.metal_move = [29, 30, 48]  # Cu, Zn, Cd
                else:
                    self.metal_move = [atomic_num-1, atomic_num, atomic_num+1, atomic_num+18]
            elif 40 <= atomic_num <= 48:  # 4d transition metals
                self.metal = atomic_num
                if atomic_num == 40:  # Zr
                    self.metal_move = [40, 41, 72, 22]  # Zr, Nb, Hf, Ti
                elif atomic_num == 48:  # Cd
                    self.metal_move = [47, 48, 80, 30]  # Ag, Cd, Hg, Zn
                else:
                    self.metal_move = [atomic_num-1, atomic_num, atomic_num+1, atomic_num+32, atomic_num-18]
            elif 72 <= atomic_num <= 80:  # 5d transition metals
                self.metal = atomic_num
                if atomic_num == 72:  # Hf
                    self.metal_move = [72, 73, 40]  # Hf, Ta, Zr
                elif atomic_num == 80:  # Hg
                    self.metal_move = [79, 80, 48]  # Au, Hg, Cd
                else:
                    self.metal_move = [atomic_num-1, atomic_num, atomic_num+1, atomic_num-32]
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
        elif self.f_block_mode == 'lanthanides_u':
            # Lanthanides + U mode: all lanthanides (Ce-Lu) plus Uranium
            lanthanides = list(range(58, 72))  # Ce (58) to Lu (71)
            allowed_elements = lanthanides + [92]  # Add U (92)

            # Start with the current element
            possible_moves = [atomic_num]

            # Add adjacent elements (±1) if they exist and are in our allowed set
            for delta in [-1, +1]:
                neighbor = atomic_num + delta
                if neighbor in allowed_elements:
                    possible_moves.append(neighbor)

            # Allow moves between lanthanides and U
            if atomic_num == 92:
                # From U, allow moves to middle lanthanides (around Nd)
                possible_moves.append(60)  # Nd
            elif atomic_num == 60:
                # From Nd, allow moves to U
                possible_moves.append(92)  # U
        elif self.f_block_mode == 'experimental':
            # Experimental mode: actinides (minus La) plus U, allowing adjacent comparisons
            lanthanides_no_la = list(range(58, 72))  # Ce (90) to La (72)
            
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
                    new_node.t_to_terminate = 30
                    new_node._determine_possible_moves()
                    expansion_list.append(new_node)
        
        # Remove identical compositions to parent (avoid going backwards)
        if self.parent is not None:
            expansion_list = [
                node for node in expansion_list 
                if node.symbols.get_chemical_formula() != self.parent.symbols.get_chemical_formula()
            ]
        
        self.expansion_list = expansion_list
        
    def rollout(self, depth: int = 1, energy_calculator=None, mode: str = 'fe', doscar_lookup=None) -> float:
        """
        Perform rollout simulation from this node.

        Args:
            depth: Number of random substitutions to perform
            energy_calculator: Energy calculator instance
            mode: Evaluation mode ('fe', 'eh', 'both', or 'weighted_alpha_beta_gamma')
            doscar_lookup: DoscarRewardLookup instance for DOSCAR rewards

        Returns:
            Reward value
        """
        import random
        
        # Create initial temporary node
        tmp_atoms = self.atoms.copy()
        tmp_metal_move = self.metal_move.copy()
        tmp_g_iv_move = self.g_iv_move.copy()
        tmp_f_block_move = getattr(self, 'f_block_move', [None])
        
        # Perform random substitutions
        for _ in range(depth):
            metal = random.choice(tmp_metal_move)
            g_iv = random.choice(tmp_g_iv_move)
            f_block = random.choice(tmp_f_block_move) if tmp_f_block_move != [None] else None
            
            # Create substituted atoms
            op_mat = []
            g_iv_list = [14, 32, 50, 82]  # Si, Ge, Sn, Pb
            f_block_list = (list(range(57, 72)) + list(range(89, 95)))  # Lanthanides + allowed actinides
            
            for atomic_num in tmp_atoms.get_atomic_numbers():
                if atomic_num in f_block_list and f_block is not None:
                    # Substitute f-block elements
                    op_mat.append(f_block - atomic_num)
                elif atomic_num in g_iv_list:
                    # Substitute Group IV elements
                    op_mat.append(g_iv - atomic_num)
                else:
                    # Substitute transition metals
                    op_mat.append(metal - atomic_num)
            
            tmp_atoms = tmp_atoms.copy()
            tmp_atoms.set_atomic_numbers(tmp_atoms.get_atomic_numbers() + op_mat)
        
        if mode == 'dos':
            # DOSCAR rewards only - no energy calculator needed
            if doscar_lookup is not None:
                formula = tmp_atoms.get_chemical_formula(mode='metal')
                doscar_reward = doscar_lookup.get_reward(formula)
                return doscar_reward
            else:
                return 0.0

        if energy_calculator is not None:
            e_form, e_above_hull = energy_calculator.calculate_energies(tmp_atoms)

            if depth == 0:
                self.e_form = e_form
                self.e_above_hull = e_above_hull

            if mode == 'eh':
                # return -(10 * e_above_hull - 0.5)
                return -e_above_hull
            elif mode == 'fe':
                return -e_form
            elif mode == 'both':
                # Legacy mode: simple sum (biased toward formation energy due to magnitude)
                return - e_form - e_above_hull
            elif mode.startswith('weighted'):
                # New weighted mode: mode='weighted_alpha_beta_gamma' where alpha, beta, gamma are the weights
                # Extract alpha, beta, gamma from mode string (e.g., 'weighted_1.0_2.0_0.5')
                try:
                    parts = mode.split('_')
                    alpha = float(parts[1])
                    beta = float(parts[2])
                    gamma = float(parts[3]) if len(parts) > 3 else 0.0
                except (IndexError, ValueError):
                    alpha = 1.0  # Default to equal weighting
                    beta = 1.0
                    gamma = 0.0

                # Get DOSCAR reward if gamma > 0 and doscar_lookup is available
                doscar_reward = 0.0
                if gamma > 0 and doscar_lookup is not None:
                    formula = tmp_atoms.get_chemical_formula(mode='metal')
                    doscar_reward = doscar_lookup.get_reward(formula)

                # Weighted combination: reward = alpha*(-e_form) + beta*(-e_above_hull) + gamma*(doscar_reward)
                # Simplifies to: reward = -alpha*e_form - beta*e_above_hull + gamma*doscar_reward
                # With typical values: e_form ~ -0.7, e_above_hull ~ 0.1, doscar_reward ~ 0.3
                # alpha=1.0, beta=1.0, gamma=1.0 gives: -1.0*(-0.7) - 1.0*(0.1) + 1.0*(0.3) = 0.7 - 0.1 + 0.3 = 0.9
                return -alpha * e_form - beta * e_above_hull + gamma * doscar_reward
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            return 0.0  # Default reward if no calculator provided
            
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
        
    def visit(self, renew_t_to_terminate: bool = False):
        """
        Update visit count and termination countdown.
        
        Args:
            renew_t_to_terminate: Whether to reset termination countdown
        """
        self.t_of_visit += 1
        
        if renew_t_to_terminate:
            self.t_to_terminate = 30
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