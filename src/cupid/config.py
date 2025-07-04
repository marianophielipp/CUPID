"""
Configuration management for CUPID.

This module provides configuration classes for different components of the CUPID system.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import torch
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy # HACK For some reason this is needed to avoid circular import issues
from lerobot.configs.policies import PreTrainedConfig
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training parameters - FIXED to match LeRobot standards
    num_steps: int = 100000  # Reduced from LeRobot's 200K for reasonable training time
    batch_size: int = 64     # LeRobot standard: 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    
    # AMP (Automatic Mixed Precision) - DISABLED for PushT like LeRobot
    use_amp: bool = False    # LeRobot disables AMP for PushT evaluation
    
    # Checkpointing
    checkpoint_every: int = 20000  # Save every 20K steps (5 checkpoints total)
    max_checkpoints: int = 5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-6
    
    @classmethod
    def for_curated(cls, num_demos: int, **kwargs) -> 'TrainingConfig':
        """Create training config adapted for curated (smaller) datasets."""
        # For curated datasets, we often need MORE training per sample, not less
        # because we have fewer samples to learn from
        base_steps = kwargs.get('num_steps', cls.num_steps)
        
        # Scale training based on dataset size - smaller datasets need more steps per sample
        if num_demos < 100:
            adapted_steps = int(base_steps * 1.5)  # 150% for very small datasets
        elif num_demos < 300:
            adapted_steps = int(base_steps * 1.2)  # 120% for small datasets
        else:
            adapted_steps = base_steps  # Same training for larger datasets
            
        # Also increase learning rate slightly for smaller datasets to learn faster
        learning_rate = kwargs.get('learning_rate', cls.learning_rate)
        if num_demos < 300:
            learning_rate = learning_rate * 1.5  # Increase LR for small datasets
            
        return cls(
            num_steps=adapted_steps,
            learning_rate=learning_rate,
            checkpoint_every=max(5000, adapted_steps // 5),  # 5 checkpoints
            **kwargs
        )


@dataclass
class PolicyConfig(DiffusionConfig):
    """
    Policy configuration, inheriting from LeRobot's DiffusionConfig.
    This ensures that we use the same battle-tested parameters and architecture
    from the original LeRobot implementation for Diffusion Policy on PushT.
    """

    # Set the policy type for LeRobot's factory
    type: str = "diffusion"
    
    # We can override or add specific parameters here if needed,
    # but for now, we rely on the defaults provided by LeRobot's DiffusionConfig,
    # which are optimized for the PushT task.
    pass


@dataclass
class InfluenceConfig:
    """Influence function computation configuration."""
    
    # Hessian computation parameters
    hessian_sample_ratio: float = 0.5   # Proportion of trajectories to use for Hessian (50%)
    min_hessian_samples: int = 50       # Minimum samples for Hessian (numerical stability)
    max_hessian_samples: int = 500      # Maximum samples for Hessian (computational efficiency)
    damping: float = 1e-3               # Damping factor for Hessian computation
    
    # Evaluation rollout parameters  
    eval_sample_ratio: float = 0.3      # Proportion of dataset for evaluation rollouts (30%)
    min_eval_samples: int = 20          # Minimum evaluation samples
    max_eval_samples: int = 200         # Maximum evaluation samples
    
    # Selection parameters  
    selection_ratio: float = 0.33       # Default to 33% based on CUPID paper findings
    min_selection: int = 10             # Minimum number of demonstrations to select
    max_selection: int = 1000           # Maximum number of demonstrations to select


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    num_episodes: int = 200  # Number of episodes for task-based evaluation (increased for better statistics)


@dataclass 
class Config:
    """Main configuration for CUPID experiments."""
    
    # Configuration identification
    config_name: str = "unknown"  # Name of the configuration for tracking
    
    # Dataset configuration
    dataset_name: str = "lerobot/pusht_image"
    max_demonstrations: Optional[int] = None  # Changed from max_episodes
    
    # Environment configuration
    environment_type: str = "cupid"  # "cupid" or "lerobot"
    lerobot_path: Optional[str] = "/home/mphielipp/robotsw/lerobot"  # Path to LeRobot installation
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Policy configuration
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    
    # Influence computation configuration
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    force_retrain: bool = False
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        # Validate environment type
        if self.environment_type not in ["cupid", "lerobot"]:
            raise ValueError(f"environment_type must be 'cupid' or 'lerobot', got '{self.environment_type}'")
        
        # Validate LeRobot path if using LeRobot environment
        if self.environment_type == "lerobot" and self.lerobot_path:
            import os
            if not os.path.exists(self.lerobot_path):
                raise ValueError(f"LeRobot path does not exist: {self.lerobot_path}")
            # HACK: Add lerobot to python path
            import sys
            if self.lerobot_path not in sys.path:
                sys.path.append(self.lerobot_path)
        
        # Convert device to torch.device if it's a string
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Config: {self.config_name}")
        logger.info(f"   Dataset: {self.dataset_name}")
        logger.info(f"   Max demonstrations: {self.max_demonstrations or 'all available'}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Training steps: {self.training.num_steps}")
        logger.info(f"   Selection ratio: {self.influence.selection_ratio:.1%}")
    
    @classmethod
    def default(cls, max_demonstrations: Optional[int] = None) -> 'Config':
        """
        Create default configuration for standard use.
        
        Args:
            max_demonstrations: Maximum number of demonstrations to use. 
                          If None, uses all available demonstrations in the dataset.
                          Default: None (use all)
        """
        return cls(
            config_name="default",
            max_demonstrations=max_demonstrations,  # Configurable, None = use all available
            training=TrainingConfig(num_steps=100000), # Increased from 20K
            policy=PolicyConfig(
                n_obs_steps=2,
                horizon=16,
                n_action_steps=8,
                num_train_timesteps=100,
            ),
            influence=InfluenceConfig(selection_ratio=0.33),
            evaluation=EvaluationConfig(num_episodes=300)  # Increased for better success rate detection
        )
    
    @classmethod
    def quick_demo(cls, max_demonstrations: Optional[int] = 1000) -> 'Config':
        """
        Create configuration for quick demonstration/testing.
        
        Args:
            max_demonstrations: Maximum number of demonstrations to use.
                         Default: 1000 for faster experimentation.
                         Set to None to use all available demonstrations.
        """
        return cls(
            config_name="quick_demo",
            max_demonstrations=max_demonstrations,   # Configurable, default 1000 for speed
            training=TrainingConfig(
                num_steps=50000,
                batch_size=64,
                checkpoint_every=10000
            ),
            policy=PolicyConfig(
                n_obs_steps=2,
                horizon=16,
                n_action_steps=8,
                num_train_timesteps=100,
                down_dims=(256, 512, 1024), # Smaller Unet for faster demo
            ),
            influence=InfluenceConfig(
                hessian_sample_ratio=0.4,
                eval_sample_ratio=0.25,
                selection_ratio=0.4
            ),
            evaluation=EvaluationConfig(num_episodes=200)
        )
    
    def get_selection_count(self, total_demos: int) -> int:
        """Calculate number of demonstrations to select based on configuration."""
        target_count = int(total_demos * self.influence.selection_ratio)
        
        # Apply min/max constraints
        target_count = max(target_count, self.influence.min_selection)
        target_count = min(target_count, self.influence.max_selection)
        
        return target_count
    
    def get_hessian_sample_count(self, total_demos: int) -> int:
        """Calculate number of trajectories to use for Hessian computation."""
        target_count = int(total_demos * self.influence.hessian_sample_ratio)
        
        # Apply min/max constraints
        target_count = max(target_count, self.influence.min_hessian_samples)
        target_count = min(target_count, self.influence.max_hessian_samples)
        
        return min(target_count, total_demos)  # Can't exceed total
    
    def get_eval_sample_count(self, total_demos: int) -> int:
        """Calculate number of trajectories to use for evaluation rollouts."""
        target_count = int(total_demos * self.influence.eval_sample_ratio)
        
        # Apply min/max constraints
        target_count = max(target_count, self.influence.min_eval_samples)
        target_count = min(target_count, self.influence.max_eval_samples)
        
        return min(target_count, total_demos)  # Can't exceed total
    
    @classmethod
    def for_demos(cls, max_demonstrations: int, **kwargs) -> 'Config':
        """
        Create configuration optimized for a specific number of demonstrations.
        
        Args:
            max_demonstrations: Number of demonstrations to use
            **kwargs: Additional parameters to override defaults in `training`,
                      `policy`, `influence`, or top-level `Config`.
            
        Examples:
            # Use 500 demonstrations with custom selection ratio
            config = Config.for_demos(500, selection_ratio=0.25)
            
            # Use 1500 demonstrations with more training steps and custom damping
            config = Config.for_demos(1500, num_steps=25000, damping=1e-2)
            
            # Use 250 demonstrations for quick testing
            config = Config.for_demos(250)
        """
        # Choose sensible defaults based on demo count
        if max_demonstrations <= 500:
            # Small dataset - more intensive training
            config = cls.quick_demo(max_demonstrations)
            config.training.num_steps = 75000
            config.influence.selection_ratio = 0.30
        elif max_demonstrations <= 1500:
            # Medium dataset - balanced approach
            config = cls.default(max_demonstrations)
            config.training.num_steps = 100000
            config.influence.selection_ratio = 0.35
        else:
            # Large dataset - standard approach
            config = cls.default(max_demonstrations)
            config.training.num_steps = 150000 # More steps for large datasets
        
        # Override any additional parameters provided from kwargs
        for key, value in kwargs.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.policy, key):
                setattr(config.policy, key, value)
            elif hasattr(config.influence, key):
                setattr(config.influence, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config

    @classmethod
    def smoke_test(cls, max_demonstrations: int = 20) -> 'Config':
        """
        FIXED: Proper smoke test configuration that actually works for diffusion models.
        
        The previous 10-step configuration was completely inadequate for diffusion policy training.
        This provides a minimal but functional test configuration.
        """
        return cls(
            config_name="smoke_test",
            dataset_name="lerobot/pusht_image",
            max_demonstrations=max_demonstrations,
            checkpoint_dir="checkpoints",
            force_retrain=True,
            training=TrainingConfig(
                num_steps=5000,
                batch_size=32,
                learning_rate=1e-3,
                checkpoint_every=2500
            ),
            policy=PolicyConfig(
                n_obs_steps=1,
                horizon=8,
                n_action_steps=4,
                num_train_timesteps=10,
                down_dims=(128, 256), # Minimal Unet
            ),
            influence=InfluenceConfig(
                hessian_sample_ratio=0.3,
                eval_sample_ratio=0.2,
                min_hessian_samples=20,
                min_eval_samples=10,
                selection_ratio=0.5
            ),
            evaluation=EvaluationConfig(num_episodes=10)
        )
    
    @classmethod
    def micro_test(cls, max_demonstrations: int = 10) -> 'Config':
        """
        Ultra-minimal test configuration for debugging the pipeline.
        Uses absolute minimum settings to test each step quickly.
        """
        return cls(
            config_name="micro_test",
            dataset_name="lerobot/pusht_image",
            max_demonstrations=max_demonstrations,
            checkpoint_dir="checkpoints",  # Use main checkpoints to reuse existing policy
            force_retrain=False,  # Don't retrain - reuse existing policy!
            training=TrainingConfig(
                num_steps=100,  # Minimal training if needed
                batch_size=8,
                learning_rate=1e-3,
                checkpoint_every=50
            ),
            policy=PolicyConfig(
                n_obs_steps=2,  # Match the small trained policy
                horizon=16,     # Match the small trained policy  
                n_action_steps=8,
                num_train_timesteps=10,
                down_dims=(256, 512), # Match the SMALL trained policy exactly
            ),
            influence=InfluenceConfig(
                hessian_sample_ratio=0.1,  # Use only 10% for Hessian
                eval_sample_ratio=0.1,     # Use only 10% for evaluation
                min_hessian_samples=5,     # Absolute minimum
                min_eval_samples=3,        # Absolute minimum
                max_hessian_samples=10,    # Cap at 10
                max_eval_samples=5,        # Cap at 5
                selection_ratio=0.5        # Select 50% of the small dataset
            ),
            evaluation=EvaluationConfig(num_episodes=3)  # Just 3 episodes
        ) 