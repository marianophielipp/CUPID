"""
Influence functions implementation for CUPID.
Handles computation of influence scores for demonstration ranking using proper Hessian-based influence functions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from PIL import Image

from .policy import DiffusionPolicy
from .config import Config

logger = logging.getLogger(__name__)


def influence_collate_fn(batch):
    """
    Custom collate function for influence computation to handle PIL images.
    
    Args:
        batch: List of sample dictionaries (batch_size=1 for influence computation)
        
    Returns:
        Collated batch dictionary with proper dimensions
    """
    # For influence computation, we process one trajectory at a time
    sample = batch[0]  # batch_size=1 for influence computation
    
    result = {}
    
    # Handle observations
    if isinstance(sample['obs'], dict):
        # Image+state observations - return as-is (lists)
        result['obs'] = sample['obs']
    else:
        # State-only observations - return as tensor without extra batch dimension
        result['obs'] = sample['obs']
    
    # Handle actions - return without extra batch dimension
    result['action'] = sample['action']
    
    # Handle reward if present
    if 'reward' in sample:
        result['reward'] = sample['reward']
    
    return result


def create_lerobot_sequence_batch(trajectory_data: List[Dict], config: Config) -> Dict[str, torch.Tensor]:
    """
    Create a proper LeRobot-style sequence batch from trajectory data.
    
    This function mimics LeRobot's delta_timestamps functionality to create
    proper observation and action sequences instead of repeating single timesteps.
    
    Args:
        trajectory_data: List of trajectory steps
        config: CUPID configuration
        
    Returns:
        Properly formatted batch for LeRobot DiffusionPolicy
    """
    if not trajectory_data:
        raise ValueError("Empty trajectory data")
    
    # Extract the sequence of states and actions
    states = []
    actions = []
    images = []
    
    for step in trajectory_data:
        if 'observation.state' in step:
            state = step['observation.state']
            # Ensure state is a numpy array
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            states.append(state)
        else:
            state = step['observation']  # fallback key
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            states.append(state)
            
        action = step['action']
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        actions.append(action)
        
        if 'observation.image' in step:
            images.append(step['observation.image'])
    
    # Convert lists to arrays with proper shape handling
    try:
        states = np.array(states, dtype=np.float32)  # Shape: (seq_len, state_dim)
        actions = np.array(actions, dtype=np.float32)  # Shape: (seq_len, action_dim)
    except ValueError as e:
        # Handle inhomogeneous shapes by taking first elements and padding/truncating
        logger.warning(f"Inhomogeneous trajectory shapes detected: {e}")
        
        # Get dimensions from first elements
        if states:
            state_dim = len(states[0]) if hasattr(states[0], '__len__') else 1
            states_fixed = []
            for state in states:
                if hasattr(state, '__len__'):
                    if len(state) >= state_dim:
                        states_fixed.append(state[:state_dim])
                    else:
                        # Pad with zeros
                        padded = np.zeros(state_dim, dtype=np.float32)
                        padded[:len(state)] = state
                        states_fixed.append(padded)
                else:
                    # Scalar state
                    padded = np.zeros(state_dim, dtype=np.float32)
                    padded[0] = state
                    states_fixed.append(padded)
            states = np.array(states_fixed, dtype=np.float32)
        
        if actions:
            action_dim = len(actions[0]) if hasattr(actions[0], '__len__') else 1
            actions_fixed = []
            for action in actions:
                if hasattr(action, '__len__'):
                    if len(action) >= action_dim:
                        actions_fixed.append(action[:action_dim])
                    else:
                        # Pad with zeros
                        padded = np.zeros(action_dim, dtype=np.float32)
                        padded[:len(action)] = action
                        actions_fixed.append(padded)
                else:
                    # Scalar action
                    padded = np.zeros(action_dim, dtype=np.float32)
                    padded[0] = action
                    actions_fixed.append(padded)
            actions = np.array(actions_fixed, dtype=np.float32)
    
    # Create observation sequence using the first n_obs_steps observations
    n_obs_steps = config.policy.n_obs_steps
    if len(states) >= n_obs_steps:
        # Use actual sequence of observations
        obs_sequence = states[:n_obs_steps]  # (n_obs_steps, state_dim)
    else:
        # Pad by repeating the first observation
        obs_sequence = np.tile(states[0], (n_obs_steps, 1))  # (n_obs_steps, state_dim)
    
    # Create action sequence using the first horizon actions
    horizon = config.policy.horizon
    if len(actions) >= horizon:
        # Use actual sequence of actions
        action_sequence = actions[:horizon]  # (horizon, action_dim)
    else:
        # Pad by repeating the last action to fill horizon
        action_sequence = np.concatenate([
            actions,
            np.tile(actions[-1], (horizon - len(actions), 1))
        ])  # (horizon, action_dim)
    
    # Convert to tensors and add batch dimension
    batch = {
        'observation.state': torch.from_numpy(obs_sequence).unsqueeze(0),  # (1, n_obs_steps, state_dim)
        'action': torch.from_numpy(action_sequence).unsqueeze(0),  # (1, horizon, action_dim)
        'action_is_pad': torch.zeros(1, horizon, dtype=torch.bool)  # (1, horizon)
    }
    
    # Handle images if present
    if images:
        # Create image sequence similar to observations
        if len(images) >= n_obs_steps:
            img_sequence = images[:n_obs_steps]
        else:
            # Repeat first image to fill n_obs_steps
            img_sequence = [images[0]] * n_obs_steps
        
        # Convert PIL images to tensors
        img_tensors = []
        for img in img_sequence:
            if isinstance(img, Image.Image):
                # Convert PIL to tensor (C, H, W)
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(img)
            else:
                # Assume already a tensor
                img_tensor = img
            img_tensors.append(img_tensor)
        
        # Stack and add batch dimension: (1, n_obs_steps, C, H, W)
        batch['observation.image'] = torch.stack(img_tensors).unsqueeze(0)
    else:
        # Add dummy images for policies that expect them
        batch['observation.image'] = torch.zeros(1, n_obs_steps, 3, 84, 84)
    
    return batch


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data with support for images."""
    def __init__(self, trajectories: List[List[Dict[str, Any]]], is_rollout: bool = False):
        self.trajectories = trajectories
        self.is_rollout = is_rollout

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trajectory = self.trajectories[idx]
        
        # For rollouts, the observation key is different
        obs_key = 'observation' if self.is_rollout else 'observation.state'
        
        # Check if we have images
        has_images = not self.is_rollout and len(trajectory) > 0 and 'observation.image' in trajectory[0]
        
        if has_images:
            # Handle image+state observations
            states = []
            images = []
            for step in trajectory:
                states.append(step['observation.state'])
                images.append(step['observation.image'])
            
            # For influence computation, we need to handle each step separately
            # So we return the trajectory as lists, not tensors
            obs = {
                'state': states,  # List of state arrays
                'image': images   # List of PIL images
            }
        else:
            # Handle state-only observations - extract as lists first
            obs_data = [step[obs_key] for step in trajectory]
            obs = obs_data  # Keep as list for now, convert in create_lerobot_sequence_batch

        # Extract actions as list first
        action_data = [step['action'] for step in trajectory]
        action = action_data  # Keep as list for now, convert in create_lerobot_sequence_batch
        
        item = {
            'obs': obs,
            'action': action,  # Keep as list
        }
        if self.is_rollout:
            # For rollouts, use the total episode reward if available, 
            # otherwise sum up step rewards, or use first step reward as fallback
            if 'total_reward' in trajectory[0]:
                item['reward'] = trajectory[0]['total_reward']
            elif len(trajectory) > 1 and all('reward' in step for step in trajectory):
                # Sum step rewards to get episode reward
                item['reward'] = sum(step['reward'] for step in trajectory)
            else:
                # Fallback to single step reward
                item['reward'] = trajectory[0].get('reward', 0.0)
        return item


class InfluenceComputer:
    """
    Computes influence scores for trajectory ranking.
    
    Uses proper Hessian-based influence functions to rank demonstrations
    by their impact on policy performance.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
    
    def compute_influence_scores(
        self,
        policy: DiffusionPolicy,
        train_trajectories: List[List[Dict]],
        eval_rollouts: List[Dict]
    ) -> np.ndarray:
        """
        Compute influence scores for each training trajectory.
        
        Args:
            policy: Trained policy
            train_trajectories: List of training trajectories
            eval_rollouts: List of evaluation rollouts
            
        Returns:
            Array of influence scores for each training trajectory
        """
        logger.info(f"Computing trajectory-based influence scores for {len(train_trajectories)} trajectories.")
        
        # Step 1: Compute Hessian inverse
        logger.info("Step 1: Computing Hessian of behavior cloning loss...")
        hessian_inv = self._compute_hessian_inverse(policy, train_trajectories)
        
        # Step 2: Compute performance gradient
        logger.info("Step 2: Computing performance gradient from rollouts...")
        perf_grad = self._compute_performance_gradient(policy, eval_rollouts)
        
        # Step 3: Compute influence for each training trajectory
        logger.info("Step 3: Computing influence score for each training trajectory...")
        influence_scores = self._compute_trajectory_influences(
            policy, train_trajectories, hessian_inv, perf_grad
        )
        
        logger.info("✅ Trajectory-based influence computation complete.")
        return influence_scores
    
    def _compute_trajectory_influences(
        self,
        policy: DiffusionPolicy,
        trajectories: List[List[Dict]],
        hessian_inv: torch.Tensor,
        perf_grad: torch.Tensor
    ) -> np.ndarray:
        """Compute influence score for each trajectory."""
        # Sample trajectories for influence computation
        config = self.config.influence
        max_trajectories = min(len(trajectories), config.max_hessian_samples or len(trajectories))
        sample_size = min(max_trajectories, int(len(trajectories) * config.hessian_sample_ratio))
        
        # Select trajectories
        if sample_size < len(trajectories):
            indices = np.random.choice(len(trajectories), sample_size, replace=False)
            selected_trajectories = [trajectories[i] for i in indices]
        else:
            selected_trajectories = trajectories
            indices = np.arange(len(trajectories))
        
        # Create dataset and dataloader
        train_dataset = TrajectoryDataset(selected_trajectories, is_rollout=False)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=influence_collate_fn)
        
        # Get all trainable parameters
        all_params = list(policy.parameters())
        
        influence_scores = []
        
        for batch in tqdm(train_loader, desc="Computing trajectory influence"):
            obs = batch['obs']
            action = batch['action']  # Already a list, no need to convert
            
            # Create proper LeRobot sequence batch from trajectory data
            if isinstance(obs, dict):
                # Convert trajectory data to proper format
                trajectory_data = []
                states = obs['state']  # List of state arrays
                images = obs['image']  # List of PIL images
                actions = action  # Already a list
                
                for i, (state, img, act) in enumerate(zip(states, images, actions)):
                    step = {
                        'observation.state': state,
                        'observation.image': img,
                        'action': act
                    }
                    trajectory_data.append(step)
                
                lerobot_batch = create_lerobot_sequence_batch(trajectory_data, self.config)
            else:
                # State-only trajectory
                trajectory_data = []
                states = obs  # Already a list
                actions = action  # Already a list
                
                for i, (state, act) in enumerate(zip(states, actions)):
                    step = {
                        'observation.state': state,
                        'action': act
                    }
                    trajectory_data.append(step)
                
                lerobot_batch = create_lerobot_sequence_batch(trajectory_data, self.config)
            
            # Move batch to device
            lerobot_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in lerobot_batch.items()}

            # Compute gradient of the loss for this training trajectory
            policy.zero_grad()
            policy.train()  # Ensure policy is in training mode
            
            # Use LeRobot's forward method which returns (loss, outputs)
            loss, _ = policy.forward(lerobot_batch)
            
            # OPTIMIZED: Get gradients more efficiently - only compute gradients for parameters that require grad
            trainable_params = [p for p in all_params if p.requires_grad]
            traj_grad = torch.autograd.grad(loss, trainable_params, retain_graph=False, allow_unused=True)
            
            # OPTIMIZED: Handle unused parameters with vectorized operations
            grad_vectors = []
            for grad in traj_grad:
                if grad is not None:
                    grad_vectors.append(grad.view(-1))
                else:
                    # Handle unused parameters - shouldn't happen with trainable_params filter
                    logger.warning("Encountered unused parameter in gradient computation")
            
            if not grad_vectors:
                logger.warning("No gradients computed for trajectory")
                influence_scores.append(0.0)
                continue
            
            traj_grad_flat = torch.cat(grad_vectors)
            
            # Compute influence: -grad_train^T * H^-1 * grad_perf
            # Since hessian_inv is diagonal, H^-1 * grad_perf is element-wise multiplication
            influence = -torch.dot(traj_grad_flat, hessian_inv * perf_grad)
            influence_scores.append(influence.item())
        
        # Convert to full-size array if we sampled
        if sample_size < len(trajectories):
            full_scores = np.zeros(len(trajectories))
            full_scores[indices] = influence_scores
            return full_scores
        else:
            return np.array(influence_scores)
    
    def _compute_performance_gradient(self, policy: DiffusionPolicy, rollouts: List[Dict]) -> torch.Tensor:
        """
        Compute the gradient of performance with respect to policy parameters.
        
        Uses the policy gradient theorem: ∇J(θ) = E[∇ log π(a|s) * R]
        """
        # Sample rollouts for performance gradient computation
        config = self.config.influence
        max_rollouts = min(len(rollouts), config.max_eval_samples or len(rollouts))
        sample_size = min(max_rollouts, int(len(rollouts) * config.eval_sample_ratio))
        
        if sample_size < len(rollouts):
            # Use indices instead of rollouts directly to avoid numpy array issues
            indices = np.random.choice(len(rollouts), sample_size, replace=False)
            sampled_rollouts = [rollouts[i] for i in indices]
        else:
            sampled_rollouts = rollouts
        
        # Convert rollouts to trajectory format for dataset
        rollout_trajectories = []
        for rollout in sampled_rollouts:
            trajectory = rollout.get('trajectory', [])
            if trajectory:
                rollout_trajectories.append(trajectory)
        
        if not rollout_trajectories:
            logger.warning("No valid trajectories found in rollouts")
            # Return zero gradient
            all_params = [p for p in policy.parameters() if p.requires_grad]
            total_params = sum(p.numel() for p in all_params)
            return torch.zeros(total_params, device=self.device)
        
        # Create dataset and dataloader for rollouts
        rollout_dataset = TrajectoryDataset(rollout_trajectories, is_rollout=True)
        rollout_loader = DataLoader(rollout_dataset, batch_size=1, shuffle=False, collate_fn=influence_collate_fn)
        
        # Get all trainable parameters
        all_params = [p for p in policy.parameters() if p.requires_grad]
        
        # Accumulate gradients
        total_grad = None
        total_weight = 0.0

        for batch in tqdm(rollout_loader, desc="Computing performance gradient"):
            obs = batch['obs']  # Already a list
            action = batch['action']  # Already a list  
            reward = batch['reward']
            
            # Handle reward - it might be a tensor or already a float
            if hasattr(reward, 'item'):
                reward = reward.item()
            
            # CRITICAL FIX: Convert reward to binary return as required by CUPID
            # R(τ) ∈ {+1, -1} based on success/failure, not actual reward values
            # For PushT task, success is typically indicated by high reward (>= 1.0)
            # or by checking if the task was completed successfully
            if reward >= 1.0:  # High reward indicates success
                binary_return = +1.0
            elif reward > 0.1:  # Some progress but not full success
                binary_return = +1.0  # Still treat as positive for CUPID
            else:  # Very low or negative reward indicates failure
                binary_return = -1.0
            
            logger.debug(f"Rollout reward: {reward:.3f} → Binary return: {binary_return:+.0f}")

            # Create proper sequence batch from rollout trajectory
            trajectory_data = []
            states = obs  # Already a list
            actions = action  # Already a list
            
            for i, (state, act) in enumerate(zip(states, actions)):
                step = {
                    'observation.state': state,
                    'action': act
                }
                trajectory_data.append(step)
            
            lerobot_batch = create_lerobot_sequence_batch(trajectory_data, self.config)
            
            # Move to device
            lerobot_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in lerobot_batch.items()}

            # The gradient of the log-probability of a trajectory is the sum of step-wise log-prob gradients.
            # ∇_θ log p(τ) = Σ_t ∇_θ log π(a_t|s_t)
            # We approximate log π(a|s) with the negative loss: -loss(s,a)
            policy.zero_grad()
            policy.train()  # Ensure policy is in training mode
            
            # Use LeRobot's forward method which returns (loss, outputs)
            loss, _ = policy.forward(lerobot_batch)
            
            # Compute gradient of negative log probability (positive gradient for likelihood)
            neg_log_prob_grad = torch.autograd.grad(-loss, all_params, retain_graph=False, allow_unused=True)
            
            # OPTIMIZED: Handle unused parameters with vectorized operations
            grad_vectors = []
            for grad in neg_log_prob_grad:
                if grad is not None:
                    grad_vectors.append(grad.view(-1))
                else:
                    logger.warning("Encountered unused parameter in performance gradient computation")
            
            if not grad_vectors:
                logger.warning("No gradients computed for rollout")
                continue
            
            rollout_grad_flat = torch.cat(grad_vectors)
            
            # Weight by binary return (CUPID requirement: R(τ) ∈ {+1, -1})
            weighted_grad = binary_return * rollout_grad_flat
            
            if total_grad is None:
                total_grad = weighted_grad
            else:
                total_grad += weighted_grad
            
            total_weight += 1.0
        
        if total_grad is None or total_weight == 0:
            logger.warning("No valid gradients computed for performance")
            total_params = sum(p.numel() for p in all_params)
            return torch.zeros(total_params, device=self.device)
        
        # Average the gradients
        perf_grad = total_grad / total_weight
        
        logger.info("✅ Computed performance gradient over {} rollouts.".format(int(total_weight)))
        return perf_grad
    
    def _compute_hessian_inverse(self, policy: DiffusionPolicy, trajectories: List[List[Dict]]) -> torch.Tensor:
        """
        Compute the inverse Hessian of the behavior cloning loss.
        
        Uses diagonal approximation for computational efficiency.
        """
        # Sample trajectories for Hessian computation
        config = self.config.influence
        max_trajectories = min(len(trajectories), config.max_hessian_samples or len(trajectories))
        sample_size = min(max_trajectories, int(len(trajectories) * config.hessian_sample_ratio))
        
        logger.info(f"Using {sample_size}/{len(trajectories)} trajectories ({sample_size/len(trajectories):.1%}) for Hessian computation")
        
        if sample_size < len(trajectories):
            # Use indices instead of trajectories directly to avoid numpy array issues
            indices = np.random.choice(len(trajectories), sample_size, replace=False)
            sampled_trajectories = [trajectories[i] for i in indices]
        else:
            sampled_trajectories = trajectories
        
        # Create dataset and dataloader
        hessian_dataset = TrajectoryDataset(sampled_trajectories, is_rollout=False)
        hessian_loader = DataLoader(hessian_dataset, batch_size=1, shuffle=False, collate_fn=influence_collate_fn)
        
        # Get all trainable parameters
        all_params = [p for p in policy.parameters() if p.requires_grad]
        trainable_params = all_params  # Already filtered
        
        # Initialize diagonal Hessian accumulator
        total_params = sum(p.numel() for p in trainable_params)
        diag_hessian = torch.zeros(total_params, device=self.device)
        
        num_samples = 0

        for batch in tqdm(hessian_loader, desc="Computing diagonal Hessian"):
            obs = batch['obs']
            action = batch['action']  # Already a list, no need to convert
            
            # Create proper LeRobot sequence batch from trajectory data
            if isinstance(obs, dict):
                # Convert trajectory data to proper format
                trajectory_data = []
                states = obs['state']  # List of state arrays
                images = obs['image']  # List of PIL images
                actions = action  # Already a list
                
                for i, (state, img, act) in enumerate(zip(states, images, actions)):
                    step = {
                        'observation.state': state,
                        'observation.image': img,
                        'action': act
                    }
                    trajectory_data.append(step)
                
                lerobot_batch = create_lerobot_sequence_batch(trajectory_data, self.config)
            else:
                # State-only trajectory
                trajectory_data = []
                states = obs  # Already a list
                actions = action  # Already a list
                
                for i, (state, act) in enumerate(zip(states, actions)):
                    step = {
                        'observation.state': state,
                        'action': act
                    }
                    trajectory_data.append(step)
                
                lerobot_batch = create_lerobot_sequence_batch(trajectory_data, self.config)
            
            # Move batch to device
            lerobot_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in lerobot_batch.items()}
            
            policy.zero_grad()
            policy.train()  # Ensure policy is in training mode
            
            # Use LeRobot's forward method which returns (loss, outputs)
            loss, _ = policy.forward(lerobot_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # OPTIMIZED: Get gradients more efficiently - only compute gradients for parameters that require grad
            grad = torch.autograd.grad(loss, trainable_params, create_graph=False, allow_unused=True)
            
            # OPTIMIZED: Handle unused parameters with vectorized operations
            grad_vectors = []
            for g in grad:
                if g is not None:
                    grad_vectors.append(g.view(-1))
                else:
                    logger.warning("Encountered unused parameter in Hessian computation")
            
            if not grad_vectors:
                logger.warning("No gradients computed for Hessian sample")
                continue
            
            grad_flat = torch.cat(grad_vectors)
            
            # Diagonal Hessian approximation: H_ii ≈ (∇_i L)^2
            diag_hessian += grad_flat ** 2
            num_samples += 1
        
        if num_samples == 0:
            logger.warning("No valid samples for Hessian computation")
            return torch.ones(total_params, device=self.device) * (1.0 / self.config.influence.damping)
        
        # Average and add damping
        diag_hessian = diag_hessian / num_samples + self.config.influence.damping
        
        # Compute inverse (element-wise for diagonal) - keep as vector, not full matrix
        hessian_inv_diag = 1.0 / diag_hessian
        
        logger.info(f"✅ Computed diagonal Hessian inverse using {num_samples} trajectories.")
        return hessian_inv_diag 

    def select_demonstrations(self, influence_scores: np.ndarray, total_demos: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select high-impact demonstrations based on influence scores.
        
        IMPROVED: Only selects demonstrations with positive influence scores,
        taking the minimum between the configured percentage and all positive scores.
        This prevents selecting demonstrations that hurt performance.
        
        Args:
            influence_scores: Array of influence scores for each demonstration
            total_demos: Total number of demonstrations available
            
        Returns:
            Tuple of (selected_indices, selected_scores)
        """
        # Calculate target number of demonstrations to select
        target_count = self.config.get_selection_count(total_demos)
        
        # Sort by influence score (highest first)
        sorted_indices = np.argsort(influence_scores)[::-1]
        sorted_scores = influence_scores[sorted_indices]
        
        # Find demonstrations with positive influence
        positive_mask = sorted_scores > 0
        num_positive = np.sum(positive_mask)
        
        if num_positive == 0:
            logger.warning("⚠️  No demonstrations with positive influence scores found!")
            logger.warning("    This suggests the influence function may not be working correctly.")
            logger.warning("    Falling back to selecting top demonstrations anyway.")
            
            # Fallback: select top demonstrations even if negative
            selected_indices = sorted_indices[:target_count]
            selected_scores = influence_scores[selected_indices]
            
            logger.info(f"Selected {len(selected_indices)} demonstrations (all negative) with influence scores "
                       f"from {selected_scores.min():.4f} to {selected_scores.max():.4f}")
        else:
            # Smart selection: only use positive influence demonstrations
            # Take minimum between target count and number of positive demonstrations
            actual_count = min(target_count, num_positive)
            
            selected_indices = sorted_indices[:actual_count]
            selected_scores = influence_scores[selected_indices]
            
            positive_percentage = (num_positive / total_demos) * 100
            target_percentage = (target_count / total_demos) * 100
            actual_percentage = (actual_count / total_demos) * 100
            
            logger.info(f"✅ Smart selection strategy applied:")
            logger.info(f"   📊 Positive influence demos: {num_positive}/{total_demos} ({positive_percentage:.1f}%)")
            logger.info(f"   🎯 Target selection: {target_count}/{total_demos} ({target_percentage:.1f}%)")
            logger.info(f"   ✅ Actual selection: {actual_count}/{total_demos} ({actual_percentage:.1f}%)")
            logger.info(f"   📈 Selected influence scores: {selected_scores.min():.4f} to {selected_scores.max():.4f}")
            
            if actual_count < target_count:
                logger.info(f"   ℹ️  Selected fewer demos than target to avoid negative influence scores")
        
        return selected_indices, selected_scores 