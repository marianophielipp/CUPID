"""
Diffusion Policy implementation for CUPID.
Handles policy architecture, creation, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from pathlib import Path
import json
import logging
import math
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ImageEncoder(nn.Module):
    """CNN encoder for processing RGB images."""
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        # CNN backbone for 96x96 RGB images
        self.conv_layers = nn.Sequential(
            # Input: (3, 96, 96)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 24, 24)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 12, 12)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (256, 6, 6)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> (256, 4, 4)
        )
        
        # Flatten and project to desired output dimension
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images: Union[torch.Tensor, list]):
        """
        Forward pass for image encoding.
        
        Args:
            images: Either tensor of shape (batch_size, 3, 96, 96) or list of PIL images
            
        Returns:
            Encoded image features of shape (batch_size, output_dim)
        """
        if isinstance(images, list):
            # Convert PIL images to tensor
            batch_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    img_tensor = self.transform(img)
                    batch_images.append(img_tensor)
                else:
                    # Assume it's already a tensor
                    batch_images.append(img)
            images = torch.stack(batch_images)
        
        # Ensure correct device
        if hasattr(self.conv_layers[0], 'weight'):
            images = images.to(self.conv_layers[0].weight.device)
        
        # CNN feature extraction
        features = self.conv_layers(images)
        
        # Flatten and project
        features = features.view(features.size(0), -1)
        encoded = self.fc(features)
        
        return encoded


class DiffusionPolicy(nn.Module):
    """
    Vision-enabled Diffusion Policy architecture for PushT task.
    
    This implements a diffusion policy with noise scheduling and U-Net architecture 
    for 2D control tasks with both image and state observations. The policy learns 
    to denoise actions conditioned on visual and proprioceptive observations.
    
    The architecture consists of:
    - Image encoder (CNN) for visual observations
    - State encoder for proprioceptive observations  
    - Time embedding for diffusion timesteps
    - Action encoder for noisy actions
    - U-Net style network with skip connections
    - DDPM sampling for action generation
    
    Args:
        config: Dictionary containing model configuration with keys:
            - input_features: Dict with observation dimensions
            - output_features: Dict with action dimensions  
            - horizon: Action sequence length (default: 16)
            - use_images: Whether to use image observations (default: True)
            
    Attributes:
        obs_dim: State observation dimension (typically 2 for PushT)
        action_dim: Action dimension (typically 2 for PushT)
        horizon: Action sequence horizon
        hidden_dim: Hidden layer dimension (256)
        num_diffusion_steps: Number of diffusion steps (100)
        use_images: Whether to process image observations
        
    Example:
        >>> config = {
        ...     'input_features': {'observation.state': {'shape': [2]}},
        ...     'output_features': {'action': {'shape': [2]}},
        ...     'horizon': 16,
        ...     'use_images': True
        ... }
        >>> policy = DiffusionPolicy(config)
        >>> obs = {'state': torch.randn(32, 2), 'image': [PIL_images]}
        >>> actions = policy.sample_action(obs)  # Sample actions
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters properly
        input_features = config.get('input_features', {})
        output_features = config.get('output_features', {})
        
        # Get observation and action dimensions
        self.obs_dim = input_features.get('observation.state', {}).get('shape', [2])[0]
        self.action_dim = output_features.get('action', {}).get('shape', [2])[0]
        self.horizon = config.get('horizon', 16)
        self.use_images = config.get('use_images', True)
        
        # Network dimensions from config
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.time_embed_dim = self.hidden_dim // 4  # e.g., 64

        # Time embedding for diffusion
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Image encoder (if using images)
        if self.use_images:
            self.image_encoder = ImageEncoder(output_dim=self.hidden_dim)
            
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Observation fusion (combine image and state if both available)
        obs_input_dim = self.hidden_dim
        if self.use_images:
            obs_input_dim = self.hidden_dim * 2  # Image + state features
            
        self.obs_fusion = nn.Sequential(
            nn.Linear(obs_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Action encoder (for noisy actions)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Main network (U-Net style with skip connections)
        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        
        for _ in range(self.num_layers // 2):
            self.down_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
            
        for _ in range(self.num_layers // 2):
            self.up_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Diffusion parameters
        self.num_diffusion_steps = config.get('num_diffusion_steps', 100)
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        
        # Precompute diffusion schedule
        self.register_buffer('betas', torch.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Action normalization parameters (for pixel coordinate datasets like PushT)
        # These will be set during training to normalize actions to [-1, 1]
        self.register_buffer('action_mean', torch.zeros(self.action_dim))
        self.register_buffer('action_std', torch.ones(self.action_dim))
        self.action_stats_initialized = False
        
    def setup_action_normalization(self, actions: torch.Tensor):
        """
        Set up action normalization parameters from training data.
        
        Args:
            actions: Training actions tensor [N, action_dim]
        """
        if not self.action_stats_initialized:
            self.action_mean = actions.mean(dim=0)
            self.action_std = actions.std(dim=0)
            # Prevent division by zero
            self.action_std = torch.clamp(self.action_std, min=1e-6)
            self.action_stats_initialized = True
            logger.info(f"Action normalization initialized:")
            logger.info(f"  Mean: {self.action_mean.cpu().numpy()}")
            logger.info(f"  Std: {self.action_std.cpu().numpy()}")
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1] range for training."""
        # Ensure action stats are on the same device as actions
        action_mean = self.action_mean.to(actions.device)
        action_std = self.action_std.to(actions.device)
        return (actions - action_mean) / action_std
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original coordinate system for inference."""
        # Ensure action stats are on the same device as actions
        action_mean = self.action_mean.to(actions.device)
        action_std = self.action_std.to(actions.device)
        return actions * action_std + action_mean

    def encode_observations(self, obs: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Encode observations (images and/or state) into feature representation.
        
        Args:
            obs: Either state tensor [batch_size, obs_dim] or dict with 'state' and optionally 'image'
            
        Returns:
            Encoded observation features [batch_size, hidden_dim]
        """
        if isinstance(obs, dict):
            # Handle dictionary input with separate image and state
            state = obs['state']
            images = obs.get('image', None)
        else:
            # Handle tensor input (state only)
            state = obs
            images = None
            
        # Encode state
        state_features = self.state_encoder(state)
        
        if self.use_images and images is not None:
            # Encode images
            image_features = self.image_encoder(images)
            
            # Combine image and state features
            combined_features = torch.cat([image_features, state_features], dim=-1)
            obs_features = self.obs_fusion(combined_features)
        else:
            # Use state features only - pass through a compatible fusion layer
            # Since obs_fusion expects either hidden_dim (state-only) or hidden_dim*2 (image+state)
            # We need to handle the state-only case properly
            if self.use_images:
                # When images are expected but not provided, pad with zeros
                batch_size = state_features.shape[0]
                zero_image_features = torch.zeros(batch_size, self.hidden_dim, device=state_features.device)
                combined_features = torch.cat([zero_image_features, state_features], dim=-1)
                obs_features = self.obs_fusion(combined_features)
            else:
                # Pure state-only mode
                obs_features = self.obs_fusion(state_features)
            
        return obs_features
        
    def forward(self, obs: Union[torch.Tensor, Dict], noisy_actions: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of diffusion policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or dict with 'state' and 'image'
            noisy_actions: Noisy action sequence [batch_size, action_dim]
            timestep: Diffusion timestep [batch_size] (optional, will use random if not provided)
            
        Returns:
            Predicted noise to subtract from noisy actions
        """
        if isinstance(obs, dict):
            batch_size = obs['state'].shape[0]
        else:
            batch_size = obs.shape[0]
        
        # Handle timestep
        if timestep is None:
            timestep = torch.randint(0, self.num_diffusion_steps, (batch_size,), 
                                   device=next(self.parameters()).device)
        
        # Ensure proper shapes
        if noisy_actions.dim() == 1:
            noisy_actions = noisy_actions.unsqueeze(0)
        if noisy_actions.shape[0] != batch_size:
            noisy_actions = noisy_actions.expand(batch_size, -1)
        
        # Encode inputs
        obs_emb = self.encode_observations(obs)
        action_emb = self.action_encoder(noisy_actions)
        time_emb = self.time_mlp(timestep.float())
        
        # Combine observation and time embeddings
        x = obs_emb + time_emb
        
        # U-Net style processing with skip connections
        skip_connections = []
        
        # Downward pass
        for layer in self.down_layers:
            x_input = torch.cat([x, action_emb], dim=-1)
            x = layer(x_input)
            skip_connections.append(x)
        
        # Upward pass with skip connections
        for i, layer in enumerate(self.up_layers):
            skip = skip_connections[-(i+1)]
            x_input = torch.cat([x, skip], dim=-1)
            x = layer(x_input)
        
        # Final output
        noise_pred = self.final_layer(x)
        
        return noise_pred
    
    def sample_action(self, obs: Union[torch.Tensor, Dict], num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample actions using DDPM sampling.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or dict with 'state' and 'image'
            num_samples: Number of action samples to generate
            temperature: Temperature for sampling (higher = more diverse actions)
            
        Returns:
            Sampled actions [batch_size, action_dim] in original coordinate system
        """
        if isinstance(obs, dict):
            batch_size = obs['state'].shape[0]
            device = obs['state'].device
        else:
            batch_size = obs.shape[0]
            device = obs.device
        
        # Start with random noise - scale by temperature for more diversity
        actions = torch.randn(batch_size, self.action_dim, device=device) * temperature
        
        # Reverse diffusion process
        for t in reversed(range(self.num_diffusion_steps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.forward(obs, actions, timestep)
            
            # Compute alpha values
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            # Compute denoising step
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
                beta_t = self.betas[t]
                
                # DDPM sampling formula
                coeff1 = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t
                coeff2 = (1 - alpha_t) * torch.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                
                actions = (actions - coeff2 * noise_pred) / torch.sqrt(alpha_t)
                
                # Add noise for non-final steps - also scale by temperature
                if t > 0:
                    noise = torch.randn_like(actions) * temperature
                    actions = actions + torch.sqrt(coeff1) * noise
            else:
                # Final step
                actions = (actions - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Denormalize actions back to original coordinate system
        if self.action_stats_initialized:
            actions = self.denormalize_actions(actions)
        
        # Clip actions to valid PushT range [0, 512] to prevent out-of-bounds actions
        actions = torch.clamp(actions, 0.0, 512.0)
        
        return actions
    
    def get_loss(self, trajectory_obs: Union[torch.Tensor, list], trajectory_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute mean diffusion loss for a trajectory.
        
        Args:
            trajectory_obs: Observation data for a trajectory (state tensor or list of dicts with 'state'/'image')
            trajectory_actions: Ground truth actions for a trajectory [num_steps, action_dim]
            
        Returns:
            Mean diffusion loss over the trajectory
        """
        num_steps = trajectory_actions.shape[0]
        device = trajectory_actions.device
        
        # Initialize action normalization if not done yet
        if not self.action_stats_initialized:
            self.setup_action_normalization(trajectory_actions)
        
        # Normalize actions to [-1, 1] range for training
        normalized_actions = self.normalize_actions(trajectory_actions)
        
        # Sample random timesteps for each step in the trajectory
        timesteps = torch.randint(0, self.num_diffusion_steps, (num_steps,), device=device)
        
        # Sample noise
        noise = torch.randn_like(normalized_actions)
        
        # Add noise to normalized actions for all steps at once
        alpha_cumprod_t = self.alphas_cumprod[timesteps]
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t).unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t).unsqueeze(-1)
        
        noisy_actions = sqrt_alpha_cumprod_t * normalized_actions + sqrt_one_minus_alpha_cumprod_t * noise
        
        # Predict noise for the entire trajectory
        noise_pred = self.forward(trajectory_obs, noisy_actions, timesteps)
        
        # Compute loss (mean over all steps)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class PolicyManager:
    """Manages policy creation, loading, and saving."""
    
    def __init__(self, config):
        """
        Initialize PolicyManager with CUPID config.
        
        Args:
            config: CUPID configuration object
        """
        from .config import Config
        self.config = config
        self.device = config.device
        self._policy_config_cache = None
    
    def load_policy_config(self) -> Dict:
        """Load policy configuration from main config."""
        if self._policy_config_cache is None:
            p_config = self.config.policy
            self._policy_config_cache = {
                'type': p_config.architecture,
                'input_features': {'observation.state': {'shape': [2]}},  # PushT state
                'output_features': {'action': {'shape': [2]}},           # PushT action
                'horizon': p_config.action_horizon,
                'hidden_dim': p_config.hidden_dim,
                'num_layers': p_config.num_layers,
                'num_diffusion_steps': p_config.num_diffusion_steps,
                'use_images': True,  # Enable image processing for vision-based tasks
            }
            logger.info(f"Loaded policy config from main config: {p_config.architecture}")
        
        return self._policy_config_cache
    
    def create_policy(self) -> DiffusionPolicy:
        """Create a fresh policy instance."""
        config = self.load_policy_config()
        return DiffusionPolicy(config).to(self.device)
    
    def save_policy(self, policy: DiffusionPolicy, filepath: Path) -> None:
        """Save policy checkpoint."""
        filepath = Path(filepath)
        logger.info(f"Saving policy: {filepath}")
        
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': self.load_policy_config()
        }, filepath)
    
    def save_policy_with_metadata(self, policy: DiffusionPolicy, filepath: Path, metadata: Dict) -> None:
        """Save policy checkpoint with training metadata."""
        import datetime
        filepath = Path(filepath)
        logger.info(f"Saving policy with metadata: {filepath}")
        
        # Add timestamp to metadata
        metadata['saved_at'] = datetime.datetime.now().isoformat()
        metadata['device'] = str(self.device)
        
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': self.load_policy_config(),
            'metadata': metadata
        }, filepath)
        
        # Log key metadata for clarity
        logger.info(f"   📋 Policy type: {metadata.get('policy_type', 'unknown')}")
        logger.info(f"   📊 Dataset: {metadata.get('num_trajectories', '?')} trajectories, {metadata.get('num_steps', '?')} steps")
        if metadata.get('selection_ratio', 1.0) < 1.0:
            logger.info(f"   🎯 Selection ratio: {metadata.get('selection_ratio', 0)*100:.1f}%")
    
    def load_policy(self, filepath: Path) -> DiffusionPolicy:
        """Load policy from checkpoint."""
        filepath = Path(filepath)
        logger.info(f"Loading policy: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Display metadata if available
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            logger.info(f"   📋 Policy type: {metadata.get('policy_type', 'unknown')}")
            logger.info(f"   📊 Dataset: {metadata.get('num_trajectories', '?')} trajectories, {metadata.get('num_steps', '?')} steps")
            logger.info(f"   📅 Saved: {metadata.get('saved_at', 'unknown')}")
            if metadata.get('selection_ratio', 1.0) < 1.0:
                logger.info(f"   🎯 Selection ratio: {metadata.get('selection_ratio', 0)*100:.1f}%")
        
        # Create policy with saved config
        config = checkpoint.get('config', self.load_policy_config())
        policy = DiffusionPolicy(config).to(self.device)
        
        # Load state dict
        policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Fix action normalization state: if action stats exist in the state dict, mark as initialized
        if hasattr(policy, 'action_mean') and hasattr(policy, 'action_std'):
            if policy.action_mean is not None and policy.action_std is not None:
                policy.action_stats_initialized = True
                logger.info(f"   🔧 Restored action normalization: mean={policy.action_mean.cpu().numpy()}, std={policy.action_std.cpu().numpy()}")
        
        return policy 