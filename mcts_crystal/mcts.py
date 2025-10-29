"""
Monte Carlo Tree Search implementation for crystal structure optimization.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .node import MCTSTreeNode


class MCTS:
    """
    Monte Carlo Tree Search algorithm for crystal structure optimization.
    """
    
    def __init__(self, root: MCTSTreeNode):
        """
        Initialize MCTS algorithm.
        
        Args:
            root: Root node of the MCTS tree
        """
        self.root = root
        self.origin_root = root
        self.current_node = root
        self.stat_dict: Dict[str, List] = {}
        self.t_warmup = len(root.metal_move) * len(root.g_iv_move)
        self.max_reward = -10.0
        self.best_node: Optional[MCTSTreeNode] = None
        self.terminated = False
        
    def select_node(self, mode: str = 'epsilon') -> List[MCTSTreeNode]:
        """
        Node selection algorithm using UCB.
        
        Args:
            mode: Selection mode ('epsilon', 'probability', 'probability_inverse', 'inverse')
            
        Returns:
            List of selected nodes for back-propagation
        """
        select_chain = [self.root]
        current = self.root
        
        while not current.expandable:
            ucb_values = []
            
            for child_node in current.children:
                if child_node.terminated:
                    ucb_values.append(-1e4)
                    if child_node.get_chemical_formula() in self.stat_dict:
                        self.stat_dict[child_node.get_chemical_formula()][2] = True
                else:
                    ucb_values.append(child_node.get_ucb())
            
            # Check if all children are terminated
            if set(ucb_values) == {-1e4}:
                self.terminated = True
                break
                
            # Select next node based on mode
            if mode == 'epsilon':
                if random.random() < 0.2:
                    current = current.children[self._probability_selector(ucb_values)]
                else:
                    current = current.children[np.argmax(ucb_values)]
            elif mode == 'probability':
                current = current.children[self._probability_selector(ucb_values)]
            elif mode == 'probability_inverse':
                current = current.children[self._probability_selector(1 - np.array(ucb_values))]
            elif mode == 'inverse':
                current = current.children[np.argmin(np.abs(ucb_values))]
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
            
    def expansion_simulation(self, rollout_depth: int = 1, n_rollout: int = 1,
                           energy_calculator=None, rollout_method: str = 'both',
                           eh_weight: float = 1.0) -> Tuple[float, bool]:
        """
        Expand selected node and perform rollout simulation.

        Args:
            rollout_depth: Depth of rollout simulation
            n_rollout: Number of rollout simulations
            energy_calculator: Energy calculator instance
            rollout_method: Rollout evaluation method ('fe', 'eh', 'both', or 'weighted')
            eh_weight: Weight for energy above hull when using 'weighted' method (default: 1.0)

        Returns:
            Tuple of (reward, renew_t_to_terminate_flag)
        """
        renew_t_to_terminate = False
        
        # Expand node if not already expanded
        if not self.current_node.children:
            if not self.current_node.expansion_list:
                self.current_node.expand()
            
        # Select random node from expansion list
        new_node = random.choice(self.current_node.expansion_list)
        
        # Try to find unexplored node (up to 10 retries)
        retry_count = 0
        while (new_node.get_chemical_formula() in self.stat_dict and 
               self.current_node != new_node and retry_count < 10):
            if retry_count == 10:
                # Use termination status from stat_dict if available
                if new_node.get_chemical_formula() in self.stat_dict:
                    new_node.terminated = self.stat_dict[new_node.get_chemical_formula()][2]
                break
            new_node = random.choice(self.current_node.expansion_list)
            retry_count += 1
            
        # Remove from expansion list and add to tree
        self.current_node.expansion_list.remove(new_node)
        new_node.add_parent(self.current_node)
        self.current_node.add_child(new_node)
        self.current_node.update_expandable()
        
        # Perform rollout simulations
        rewards = []

        if rollout_method == 'fe':
            # Formation energy only
            rewards.append(new_node.rollout(depth=0, energy_calculator=energy_calculator, mode='fe'))
            for _ in range(n_rollout - 1):
                rollout_reward = new_node.rollout(
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode='fe'
                )
                rewards.append((1 - 0.1 * rollout_depth) * rollout_reward)
        elif rollout_method == 'eh':
            # Energy above hull only
            rewards.append(new_node.rollout(depth=0, energy_calculator=energy_calculator, mode='eh'))
            for _ in range(n_rollout - 1):
                rollout_reward = new_node.rollout(
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode='eh'
                )
                rewards.append((1 - 0.1 * rollout_depth) * rollout_reward)
        elif rollout_method == 'weighted':
            # Weighted combination of formation energy and energy above hull
            weighted_mode = f'weighted_{eh_weight}'
            rewards.append(new_node.rollout(depth=0, energy_calculator=energy_calculator, mode=weighted_mode))
            for _ in range(n_rollout - 1):
                rollout_reward = new_node.rollout(
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode=weighted_mode
                )
                rewards.append((1 - 0.1 * rollout_depth) * rollout_reward)
        else:  # rollout_method == 'both'
            # Both formation energy and energy above hull (original behavior)
            rewards.append(new_node.rollout(depth=0, energy_calculator=energy_calculator, mode='fe'))
            for _ in range(n_rollout - 1):
                rollout_reward = new_node.rollout(
                    depth=rollout_depth,
                    energy_calculator=energy_calculator,
                    mode='eh'
                )
                rewards.append((1 - 0.1 * rollout_depth) * rollout_reward)
            
        reward = np.max(rewards)
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
            self.stat_dict[formula] = [
                self.current_node.get_rewards(total=False),
                0,
                False,
                self.current_node.e_above_hull,
                self.current_node.e_form
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
        
    def run(self, n_iterations: int, energy_calculator=None,
            rollout_depth: int = 1, n_rollout: int = 10,
            selection_mode: str = 'epsilon', rollout_method: str = 'both',
            eh_weight: float = 1.0) -> Dict:
        """
        Run MCTS algorithm for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations
            energy_calculator: Energy calculator instance
            rollout_depth: Depth of rollout simulations
            n_rollout: Number of rollout simulations per expansion
            selection_mode: Node selection mode
            rollout_method: Rollout evaluation method ('fe', 'eh', 'both', or 'weighted')
            eh_weight: Weight for energy above hull when using 'weighted' method (default: 1.0)
                      Higher values prioritize hull stability over formation energy

        Returns:
            Dictionary containing run statistics
        """
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
                eh_weight=eh_weight
            )
            
            # Back-propagation
            self.back_propagation(reward, select_chain, renew_t_to_terminate)
            
            # Update statistics
            self.stat_node_visited()
            
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
        stat_df = pd.DataFrame(self.stat_dict).T
        if not stat_df.empty:
            stat_df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
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