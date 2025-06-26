"""
Training module for CUPID.
Handles policy training with proper diffusion policy training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from .policy import DiffusionPolicy, PolicyManager

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data."""
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.data[idx]
        return {
            'obs': np.array(sample['observation.state'], dtype=np.float32),
            'action': np.array(sample['action'], dtype=np.float32)
        }


class PolicyTrainer:
    """Handles training of diffusion policies."""
    
    def __init__(self, config):
        """
        Initialize PolicyTrainer with CUPID config.
        
        Args:
            config: CUPID configuration object
        """
        self.config = config
        self.device = config.device
    
    def train_policy(
        self,
        dataset: List[Dict],
        policy_manager: PolicyManager
    ) -> Tuple[DiffusionPolicy, List[float]]:
        """
        Train a diffusion policy on a given set of demonstrations.
        
        Args:
            dataset: The dataset to train on (a flat list of steps).
            policy_manager: PolicyManager instance.
            
        Returns:
            Tuple of (trained_policy, loss_history)
        """
        training_config = self.config.training
        
        logger.info(f"Starting training: {training_config.num_steps} steps, LR={training_config.learning_rate}")
        
        train_dataset = TrajectoryDataset(dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=4,  # Use multiple workers for data loading
            pin_memory=True
        )
        
        # Calculate training parameters
        steps_per_epoch = len(train_loader)
        total_epochs = (training_config.num_steps + steps_per_epoch - 1) // steps_per_epoch
        
        logger.info(f"Training for {training_config.num_steps} steps (~{total_epochs} epochs)")
        logger.info(f"Batch size: {training_config.batch_size}, Steps per epoch: {steps_per_epoch}")
        
        # Create policy and optimizer
        policy = policy_manager.create_policy()
        optimizer = optim.Adam(
            policy.parameters(), 
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        loss_history = []
        
        # Training loop
        step = 0
        epoch = 0
        
        with tqdm(total=training_config.num_steps, desc="Training") as pbar:
            while step < training_config.num_steps:
                epoch += 1
                epoch_losses = []
                
                for batch in train_loader:
                    if step >= training_config.num_steps:
                        break
                    
                    # Data is already batched and on its way to the device
                    obs_tensor = batch['obs'].to(self.device)
                    action_tensor = batch['action'].to(self.device)
                    
                    # Forward pass with diffusion loss
                    optimizer.zero_grad()
                    
                    # Use the policy's built-in diffusion loss
                    loss = policy.get_loss(obs_tensor, action_tensor)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Record loss
                    loss_value = loss.item()
                    epoch_losses.append(loss_value)
                    loss_history.append(loss_value)
                    
                    step += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss_value:.6f}',
                        'epoch': epoch
                    })
                    
                    # Log progress periodically
                    if step % 1000 == 0:
                        avg_loss = np.mean(epoch_losses[-min(10, len(epoch_losses)):])
                        logger.info(f"Step {step}/{training_config.num_steps}, Recent loss: {avg_loss:.6f}")
                
                # Log epoch summary
                if epoch_losses:
                    avg_epoch_loss = np.mean(epoch_losses)
                    logger.info(f"Epoch {epoch} completed. Avg loss: {avg_epoch_loss:.6f}")
        
        final_loss = loss_history[-1] if loss_history else 0.0
        logger.info(f"Training completed. Final loss: {final_loss:.6f}")
        
        # Log training summary
        if len(loss_history) > 100:
            initial_loss = np.mean(loss_history[:10])
            final_loss_avg = np.mean(loss_history[-10:])
            improvement = ((initial_loss - final_loss_avg) / initial_loss) * 100
            logger.info(f"Training improvement: {improvement:.1f}% (from {initial_loss:.6f} to {final_loss_avg:.6f})")
        
        return policy, loss_history
    
    def evaluate_policy(
        self,
        policy: DiffusionPolicy,
        dataset: List[Dict],
        num_samples: int = 200
    ) -> Dict[str, float]:
        """
        Evaluate a policy on dataset samples using proper diffusion sampling.
        
        Args:
            policy: Policy to evaluate
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating policy on {num_samples} samples")
        
        policy.eval()
        losses = []
        action_accuracies = []
        
        # Create a loader for evaluation
        eval_dataset = TrajectoryDataset(dataset)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        num_evaluated = 0
        with torch.no_grad():
            for batch in eval_loader:
                if num_evaluated >= num_samples:
                    break
                
                # Prepare data
                obs = batch['obs'].to(self.device)
                action_gt = batch['action'].to(self.device)
                
                # Compute diffusion loss
                loss = policy.get_loss(obs, action_gt)
                losses.extend([loss.item()] * obs.size(0))
                
                # Sample action and compare with ground truth
                action_pred = policy.sample_action(obs)
                action_error = torch.mean((action_pred - action_gt) ** 2, dim=1)
                action_accuracy = 1.0 / (1.0 + action_error)
                action_accuracies.extend(action_accuracy.cpu().numpy())
                
                num_evaluated += obs.size(0)
        
        metrics = {
            'avg_loss': np.mean(losses[:num_samples]),
            'std_loss': np.std(losses[:num_samples]),
            'min_loss': np.min(losses[:num_samples]),
            'max_loss': np.max(losses[:num_samples]),
            'action_accuracy': np.mean(action_accuracies[:num_samples]),
            'action_accuracy_std': np.std(action_accuracies[:num_samples])
        }
        
        logger.info(f"Evaluation complete. Avg loss: {metrics['avg_loss']:.6f}, Action accuracy: {metrics['action_accuracy']:.3f}")
        
        policy.train()
        return metrics 