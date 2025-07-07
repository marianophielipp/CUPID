"""
Policy management for CUPID, integrating LeRobot's DiffusionPolicy.
Handles policy creation, and checkpoint management by leveraging the LeRobot library.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import json
import logging

# HACK to be ablep to import DiffusionConfig even if it's not used
from cupid.config import PolicyConfig

# LeRobot imports for policy creation and management
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.factory import make_policy, get_policy_class
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.utils import get_safe_torch_device

logger = logging.getLogger(__name__)


class PolicyManager:
    """
    Manages the lifecycle of a LeRobot DiffusionPolicy, including creation,
    saving, and loading.
    """

    def __init__(self, config: 'Config', dataset_metadata: LeRobotDatasetMetadata):
        """
        Initialize the PolicyManager.

        Args:
            config: The main CUPID configuration object.
            dataset_metadata: LeRobot dataset metadata containing information 
                              about observation and action spaces.
        """
        self.config = config
        self.policy_config = config.policy
        self.dataset_metadata = dataset_metadata

    def create_policy(self) -> DiffusionPolicy:
        """
        Create a new DiffusionPolicy instance using the LeRobot factory.
        This ensures that the policy is instantiated with the correct
        architecture and hyperparameters as defined in the config.
        """
        logger.info("Creating LeRobot DiffusionPolicy...")

        # Use LeRobot's make_policy function with dataset metadata
        policy = make_policy(self.policy_config, ds_meta=self.dataset_metadata)

        # Get device from policy or config
        device = getattr(policy, 'device', self.config.device)
        logger.info(f"LeRobot DiffusionPolicy created and moved to {device}.")
        logger.info(f"   - Vision Backbone: {policy.config.vision_backbone}")
        logger.info(f"   - U-Net Down Dims: {policy.config.down_dims}")
        logger.info(f"   - Action Horizon: {policy.config.horizon}")
        logger.info(f"   - Obs Steps: {policy.config.n_obs_steps}")

        return policy

    def save_policy(self, policy: torch.nn.Module, filepath: Path) -> None:
        """
        Save policy to checkpoint file using PyTorch's standard approach.
        
        Args:
            policy: Policy model to save
            filepath: Path to save checkpoint
        """
        logger.info(f"Saving policy to {filepath}...")
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using PyTorch's standard approach
        checkpoint = {
            'model_state_dict': policy.state_dict(),
            'config': self.policy_config,
            'dataset_metadata': self.dataset_metadata
        }
        
        torch.save(checkpoint, str(filepath))
        logger.info(f"Policy saved successfully to {filepath}")

    def load_policy(self, filepath: Path) -> torch.nn.Module:
        """
        Load policy from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Loaded policy model
        """
        logger.info(f"📂 Loading policy from {filepath}...")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(str(filepath), map_location=self.config.device)
        
        # Create policy
        policy = make_policy(self.policy_config, ds_meta=self.dataset_metadata)
        
        # Load state dict
        policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Get device from policy or config
        device = getattr(policy, 'device', self.config.device)
        logger.info(f"Policy loaded successfully and moved to {device}.")
        return policy 