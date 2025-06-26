"""
Influence functions implementation for CUPID.
Handles computation of influence scores for demonstration ranking using proper Hessian-based influence functions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

from .policy import DiffusionPolicy
from .config import Config

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data."""
    def __init__(self, trajectories: List[List[Dict[str, Any]]], is_rollout: bool = False):
        self.trajectories = trajectories
        self.is_rollout = is_rollout

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trajectory = self.trajectories[idx]
        
        # For rollouts, the observation key is different
        obs_key = 'observation' if self.is_rollout else 'observation.state'

        obs = np.array([step[obs_key] for step in trajectory], dtype=np.float32)
        action = np.array([step['action'] for step in trajectory], dtype=np.float32)
        
        item = {
            'obs': torch.from_numpy(obs),
            'action': torch.from_numpy(action),
        }
        if self.is_rollout:
            item['reward'] = trajectory[0].get('reward', 0.0)
        return item


class InfluenceComputer:
    """Computes influence scores for demonstration ranking using proper CUPID influence functions."""
    
    def __init__(self, config: Config):
        """
        Initialize InfluenceComputer with CUPID config.
        
        Args:
            config: CUPID configuration object
        """
        self.config = config
        self.device = config.device
        self.damping = config.influence.damping
    
    def compute_influence_scores(
        self,
        policy: DiffusionPolicy,
        train_trajectories: List[List[Dict]],
        eval_rollouts: List[Dict]
    ) -> np.ndarray:
        """
        Compute proper CUPID influence scores using trajectory-based influence functions.
        
        The influence of a training trajectory (ξ) on policy performance is:
        Ψ_inf(ξ) ≈ - ∇_θ J(π_θ)^T H_bc^(-1) ∇_θ ℓ_traj(ξ)
        
        where:
        - ∇_θ J(π_θ) is the performance gradient, estimated from eval_rollouts.
        - H_bc is the Hessian of the behavior cloning loss.
        - ∇_θ ℓ_traj(ξ) is the gradient of the loss for a single training trajectory.
        
        Args:
            policy: Trained baseline policy.
            train_trajectories: List of training trajectories.
            eval_rollouts: List of policy rollouts for performance evaluation.
            
        Returns:
            Array of influence scores for each training trajectory.
        """
        logger.info(f"Computing trajectory-based influence scores for {len(train_trajectories)} trajectories.")
        
        policy.eval()
        
        # Step 1: Compute Hessian of behavior cloning loss using training trajectories.
        logger.info("Step 1: Computing Hessian of behavior cloning loss...")
        hessian_inv_diag = self._compute_hessian_inverse(policy, train_trajectories)
        
        # Step 2: Compute performance gradient from evaluation rollouts.
        logger.info("Step 2: Computing performance gradient from rollouts...")
        performance_gradient = self._compute_performance_gradient(policy, eval_rollouts)
        
        # Step 3: Compute influence for each training trajectory.
        logger.info("Step 3: Computing influence score for each training trajectory...")
        
        influence_scores = []
        train_dataset = TrajectoryDataset(train_trajectories)
        train_loader = DataLoader(train_dataset, batch_size=1) # Process one trajectory at a time

        for batch in tqdm(train_loader, desc="Computing trajectory influence"):
            obs = batch['obs'][0].to(self.device)
            action = batch['action'][0].to(self.device)

            # Compute gradient of the loss for this training trajectory
            policy.zero_grad()
            loss = policy.get_loss(obs, action)
            traj_grad = torch.autograd.grad(loss, policy.parameters(), retain_graph=False)
            traj_grad_vector = torch.cat([g.view(-1) for g in traj_grad if g is not None])

            # Compute influence: - (perf_grad^T * H_inv * traj_grad)
            influence = -torch.sum(performance_gradient * hessian_inv_diag * traj_grad_vector)
            influence_scores.append(influence.item())
            
        influence_scores = np.array(influence_scores)
        if influence_scores.max() > influence_scores.min():
            influence_scores = (influence_scores - influence_scores.min()) / (influence_scores.max() - influence_scores.min())
        
        logger.info("✅ Trajectory-based influence computation complete.")
        return influence_scores

    def _compute_performance_gradient(self, policy: DiffusionPolicy, rollouts: List[Dict]) -> torch.Tensor:
        """Computes the REINFORCE-style performance gradient from evaluation rollouts."""
        if not rollouts:
            raise ValueError("Evaluation rollouts cannot be empty for performance gradient computation.")

        # Re-format rollouts to be a list of trajectories (list of steps)
        eval_trajectories = [r['trajectory'] for r in rollouts]
        
        # Add the episode reward to each step for easy access in the dataset
        for i, traj in enumerate(eval_trajectories):
            reward = rollouts[i].get('reward', 0.0)
            for step in traj:
                step['reward'] = reward

        total_reward_weighted_grad = None
        
        rollout_dataset = TrajectoryDataset(eval_trajectories, is_rollout=True)
        rollout_loader = DataLoader(rollout_dataset, batch_size=1)

        for batch in tqdm(rollout_loader, desc="Computing performance gradient"):
            obs = batch['obs'][0].to(self.device)
            action = batch['action'][0].to(self.device)
            reward = batch['reward'].item()

            # The gradient of the log-probability of a trajectory is the sum of step-wise log-prob gradients.
            # ∇_θ log p(τ) = Σ_t ∇_θ log π(a_t|s_t)
            # We approximate log π(a|s) with the negative loss: -loss(s,a)
            policy.zero_grad()
            loss = policy.get_loss(obs, action)
            
            # We want gradient of log-prob, which is grad of -loss.
            traj_log_prob_grad = torch.autograd.grad(-loss, policy.parameters(), retain_graph=False)
            grad_vector = torch.cat([g.view(-1) for g in traj_log_prob_grad if g is not None])
            
            if total_reward_weighted_grad is None:
                total_reward_weighted_grad = torch.zeros_like(grad_vector)

            total_reward_weighted_grad += grad_vector * reward
        
        # Average over the number of rollouts
        performance_gradient = total_reward_weighted_grad / len(rollouts)
        logger.info(f"✅ Computed performance gradient over {len(rollouts)} rollouts.")
        return performance_gradient

    def _compute_hessian_inverse(self, policy: DiffusionPolicy, trajectories: List[List[Dict]]) -> torch.Tensor:
        """Computes diagonal approximation of inverse Hessian using trajectories."""
        n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        hessian_diag = torch.zeros(n_params, device=self.device)
        
        # Use a subset of trajectories for Hessian computation for efficiency
        num_hessian_samples = min(len(trajectories), self.config.influence.num_samples)
        sample_indices = np.random.choice(len(trajectories), num_hessian_samples, replace=False)
        hessian_trajectories = [trajectories[i] for i in sample_indices]
        
        hessian_dataset = TrajectoryDataset(hessian_trajectories)
        hessian_loader = DataLoader(hessian_dataset, batch_size=1)

        for batch in tqdm(hessian_loader, desc="Computing diagonal Hessian"):
            obs = batch['obs'][0].to(self.device)
            action = batch['action'][0].to(self.device)
            
            policy.zero_grad()
            loss = policy.get_loss(obs, action)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            grad = torch.autograd.grad(loss, policy.parameters(), create_graph=False)
            grad_vector = torch.cat([g.view(-1) for g in grad if g is not None])
            
            hessian_diag += grad_vector ** 2
        
        hessian_diag /= num_hessian_samples
        hessian_diag += self.damping
        
        # Handle numerical issues
        if torch.any(hessian_diag <= 0):
            hessian_diag = torch.clamp(hessian_diag, min=self.damping * 10)
        
        hessian_inv_diag = 1.0 / hessian_diag
        
        logger.info(f"✅ Computed diagonal Hessian inverse using {num_hessian_samples} trajectories.")
        return hessian_inv_diag
    
    def select_demonstrations(
        self,
        influence_scores: np.ndarray,
        dataset_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select demonstrations by influence scores using configured selection ratio.
        
        Args:
            influence_scores: Array of influence scores for each trajectory
            dataset_size: Total number of trajectories
            
        Returns:
            Tuple of (selected_indices, selected_scores)
        """
        num_to_select = self.config.get_selection_count(dataset_size)
        selection_ratio = self.config.influence.selection_ratio
        
        logger.info(f"Selecting {num_to_select} demonstrations ({selection_ratio*100:.1f}% of {dataset_size})")
        
        # Sort by influence score (descending - higher influence is better)
        sorted_indices = np.argsort(influence_scores)[::-1]
        
        selected_indices = sorted_indices[:num_to_select]
        selected_scores = influence_scores[selected_indices]
        
        logger.info(f"Selected {len(selected_indices)} demonstrations with score range: [{selected_scores.min():.4f}, {selected_scores.max():.4f}]")
        return selected_indices, selected_scores 