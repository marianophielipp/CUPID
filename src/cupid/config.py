"""
Configuration management for CUPID.

This module provides configuration classes for different components of the CUPID system.
"""

from dataclasses import dataclass
from typing import Optional, Union
import torch


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training parameters
    num_steps: int = 20000  # LeRobot standard: 20K steps
    batch_size: int = 64    # LeRobot standard: 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    
    # Checkpointing
    checkpoint_every: int = 5000  # Save every 5K steps (4 checkpoints total)
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
            checkpoint_every=max(1000, adapted_steps // 4),  # 4 checkpoints
            **kwargs
        )


@dataclass
class PolicyConfig:
    """Policy architecture configuration."""
    
    # Architecture
    architecture: str = "DiffusionPolicy"
    
    # Network dimensions
    hidden_dim: int = 256
    num_layers: int = 4
    
    # Diffusion-specific parameters
    num_diffusion_steps: int = 100
    noise_schedule: str = "cosine"
    
    # Action prediction
    action_horizon: int = 8
    observation_horizon: int = 2


@dataclass
class InfluenceConfig:
    """Influence function computation configuration."""
    
    # Sampling parameters
    num_samples: int = 200  # Number of rollout samples for influence computation
    damping: float = 1e-3   # Damping factor for Hessian computation
    
    # Selection parameters  
    selection_ratio: float = 0.33  # Default to 33% based on CUPID paper findings
    min_selection: int = 10        # Minimum number of demonstrations to select
    max_selection: int = 1000      # Maximum number of demonstrations to select


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    num_episodes: int = 50  # Number of episodes for task-based evaluation


@dataclass 
class Config:
    """Main CUPID configuration."""
    
    # Dataset
    dataset_name: str = "lerobot/pusht"
    max_episodes: Optional[int] = None
    
    # Device
    device: Union[str, torch.device] = "auto"
    
    # Training control
    force_retrain: bool = False  # Whether to force retraining even if checkpoints exist
    
    # Component configurations
    training: TrainingConfig = None
    policy: PolicyConfig = None
    influence: InfluenceConfig = None
    evaluation: EvaluationConfig = None
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.training is None:
            self.training = TrainingConfig()
        if self.policy is None:
            self.policy = PolicyConfig()
        if self.influence is None:
            self.influence = InfluenceConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
            
        # Handle device auto-detection
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
    
    @classmethod
    def default(cls, max_episodes: Optional[int] = None) -> 'Config':
        """
        Create default configuration for standard use.
        
        Args:
            max_episodes: Maximum number of demonstrations to use. 
                         If None, uses all available demonstrations in the dataset.
                         Default: None (use all)
        """
        return cls(
            max_episodes=max_episodes,  # Configurable, None = use all available
            training=TrainingConfig(num_steps=20000),
            influence=InfluenceConfig(selection_ratio=0.33),
            evaluation=EvaluationConfig(num_episodes=100)
        )
    
    @classmethod
    def quick_demo(cls, max_episodes: Optional[int] = 1000) -> 'Config':
        """
        Create configuration for quick demonstration/testing.
        
        Args:
            max_episodes: Maximum number of demonstrations to use.
                         Default: 1000 for faster experimentation.
                         Set to None to use all available demonstrations.
        """
        return cls(
            max_episodes=max_episodes,   # Configurable, default 1000 for speed
            training=TrainingConfig(
                num_steps=15000,     # Good balance of quality and speed
                batch_size=64,       # Standard batch size
                checkpoint_every=3000  # Adjusted for step count
            ),
            influence=InfluenceConfig(
                num_samples=200,     # Reasonable for speed
                selection_ratio=0.4  # 40% selection ratio
            ),
            evaluation=EvaluationConfig(num_episodes=50)
        )
    
    def get_selection_count(self, total_demos: int) -> int:
        """Calculate number of demonstrations to select based on configuration."""
        target_count = int(total_demos * self.influence.selection_ratio)
        
        # Apply min/max constraints
        target_count = max(target_count, self.influence.min_selection)
        target_count = min(target_count, self.influence.max_selection)
        target_count = min(target_count, total_demos)  # Can't select more than available
        
        return target_count
    
    @classmethod
    def for_demos(cls, max_episodes: int, **kwargs) -> 'Config':
        """
        Create configuration optimized for a specific number of demonstrations.
        
        Args:
            max_episodes: Number of demonstrations to use
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
        if max_episodes <= 500:
            # Small dataset - need more intensive training
            config = cls.quick_demo(max_episodes)
            config.training.num_steps = 18000
            config.influence.selection_ratio = 0.30
        elif max_episodes <= 1500:
            # Medium dataset - balanced approach
            config = cls.default(max_episodes)
            config.training.num_steps = 18000
            config.influence.selection_ratio = 0.35
        else:
            # Large dataset - standard approach
            config = cls.default(max_episodes)
        
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
    def smoke_test(cls, max_episodes: int = 20) -> 'Config':
        """Ultra-fast smoke test configuration for validating pipeline execution."""
        return cls(
            dataset_name="lerobot/pusht",
            max_episodes=max_episodes,
            checkpoint_dir="checkpoints_debug",
            force_retrain=True,
            training=TrainingConfig(
                num_steps=10,      # Minimal steps
                batch_size=8,
                learning_rate=1e-3,
                checkpoint_every=10
            ),
            influence=InfluenceConfig(
                num_samples=2,       # Minimal samples for Hessian/rollouts
                damping=1e-3,
                selection_ratio=0.5 # Ensure a few demos are selected
            ),
            evaluation=EvaluationConfig(num_episodes=2) # Minimal episodes
        ) 