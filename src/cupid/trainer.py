"""
Training module for CUPID.
Handles policy training with proper diffusion policy training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from PIL import Image
from torch.optim import AdamW
from transformers import get_scheduler
import torchvision

from .policy import DiffusionPolicy, PolicyManager
from .config import Config

logger = logging.getLogger(__name__)


def create_custom_collate_fn(horizon, n_obs_steps=2):
    """
    Create a custom collate function with the specified horizon and observation steps.
    
    Args:
        horizon: The action horizon to use for creating action sequences
        n_obs_steps: Number of observation steps expected by the policy
    
    Returns:
        A collate function that uses the specified horizon and n_obs_steps
    """
    def custom_collate_fn(batch):
        """
        Custom collate function for trajectory batches with PIL images.
        
        OPTIMIZED: Using list comprehensions and vectorized operations.
        Formats data to match LeRobot's expected input format with action sequences.
        """
        # Vectorized extraction using list comprehensions with LeRobot key names
        states = [sample['observation.state'] for sample in batch]
        actions = [sample['action'] for sample in batch]
        
        # Create action sequences by repeating the current action
        # This is a simplification - ideally we'd use actual future actions from trajectories
        action_sequences = []
        for action in actions:
            # Repeat the action 'horizon' times to create a sequence
            action_seq = np.tile(action, (horizon, 1))  # Shape: (horizon, action_dim)
            action_sequences.append(action_seq)
        
        # Create observation sequences by repeating the current observation
        # For n_obs_steps=2, we duplicate the current observation
        state_sequences = []
        for state in states:
            # Repeat the state 'n_obs_steps' times to create a sequence
            state_seq = np.tile(state, (n_obs_steps, 1))  # Shape: (n_obs_steps, state_dim)
            state_sequences.append(state_seq)
        
        # Stack tensors efficiently
        result = {
            'observation.state': torch.stack([torch.from_numpy(s) for s in state_sequences]),  # (B, n_obs_steps, state_dim)
            'action': torch.stack([torch.from_numpy(a) for a in action_sequences]),  # (B, horizon, action_dim)
            'action_is_pad': torch.zeros(len(batch), horizon, dtype=torch.bool)  # (B, horizon) - no padding
        }
        
        # Handle images efficiently (only extract if any sample has images)
        if batch and 'observation.image' in batch[0]:
            # Convert PIL images to tensors and stack them
            images = []
            for sample in batch:
                # Convert PIL image to tensor
                if isinstance(sample['observation.image'], Image.Image):
                    # Convert PIL to tensor (C, H, W) and normalize to [0, 1]
                    img_tensor = torchvision.transforms.functional.to_tensor(sample['observation.image'])
                    # Repeat for n_obs_steps: (n_obs_steps, C, H, W)
                    img_seq = img_tensor.unsqueeze(0).repeat(n_obs_steps, 1, 1, 1)
                    images.append(img_seq)
                else:
                    # Already a tensor, repeat for n_obs_steps
                    img_tensor = sample['observation.image']
                    if len(img_tensor.shape) == 3:  # (C, H, W)
                        img_seq = img_tensor.unsqueeze(0).repeat(n_obs_steps, 1, 1, 1)  # (n_obs_steps, C, H, W)
                    else:
                        img_seq = img_tensor
                    images.append(img_seq)
            result['observation.image'] = torch.stack(images)  # (B, n_obs_steps, C, H, W)
        
        return result

    return custom_collate_fn


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data with image and state observations."""
    def __init__(self, data: List[Dict], use_images: bool = True):
        self.data = data
        self.use_images = use_images

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, Image.Image]]:
        sample = self.data[idx]
        
        result = {
            'observation.state': np.array(sample['observation.state'], dtype=np.float32),
            'action': np.array(sample['action'], dtype=np.float32)
        }
        
        # Include image if available and requested
        if self.use_images and 'observation.image' in sample:
            result['observation.image'] = sample['observation.image']  # Keep as PIL Image
            
        return result


class Trainer:
    """Handles policy training, evaluation, and checkpointing."""

    def __init__(self, config: Config, dataloader: DataLoader, policy_manager: PolicyManager):
        self.config = config
        self.dataloader = dataloader
        self.policy_manager = policy_manager

    def train_policy(self, policy, is_curated: bool = False):
        """
        Train a policy (either baseline or curated).

        Args:
            policy: The policy model to train.
            is_curated: Flag to indicate if this is the curated policy.

        Returns:
            The trained policy and its loss history.
        """
        training_config = self.config.training
        
        optimizer = AdamW(
            policy.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=training_config.num_steps,
        )

        logger.info(f"🚀 Starting training for {training_config.num_steps} steps...")
        policy.train()
        loss_history = []
        
        progress_bar = tqdm(range(training_config.num_steps), desc=f"Training {'Curated' if is_curated else 'Baseline'} Policy")

        step = 0
        while step < training_config.num_steps:
            for batch in self.dataloader:
                if step >= training_config.num_steps:
                    break
                    
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get loss from LeRobot policy using forward method
                loss, _ = policy.forward(batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_history.append(loss.item())
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                step += 1
        
        progress_bar.close()
        logger.info("✅ Training complete.")
        return policy, loss_history

    def evaluate_policy(self, policy) -> Dict[str, float]:
        """
        DEPRECATED: This method has been removed. 
        Use TaskEvaluator.evaluate_policy_on_task() for proper task-based evaluation.
        """
        logger.error("❌ Trainer.evaluate_policy() is deprecated and broken!")
        logger.error("   Use TaskEvaluator.evaluate_policy_on_task() instead")
        raise NotImplementedError("Use TaskEvaluator.evaluate_policy_on_task() for proper evaluation") 