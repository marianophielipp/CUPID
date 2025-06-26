"""
Diffusion Policy implementation for CUPID.
Handles policy architecture, creation, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path
import json
import logging
import math

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


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy architecture for PushT task.
    
    This implements a diffusion policy with noise scheduling and U-Net architecture 
    for 2D control tasks. The policy learns to denoise actions conditioned on 
    observations using a diffusion process.
    
    The architecture consists of:
    - Time embedding for diffusion timesteps
    - Observation encoder
    - Action encoder for noisy actions
    - U-Net style network with skip connections
    - DDPM sampling for action generation
    
    Args:
        config: Dictionary containing model configuration with keys:
            - input_features: Dict with observation dimensions
            - output_features: Dict with action dimensions  
            - horizon: Action sequence length (default: 16)
            
    Attributes:
        obs_dim: Observation dimension (typically 2 for PushT)
        action_dim: Action dimension (typically 2 for PushT)
        horizon: Action sequence horizon
        hidden_dim: Hidden layer dimension (256)
        num_diffusion_steps: Number of diffusion steps (100)
        
    Example:
        >>> config = {
        ...     'input_features': {'observation.state': {'shape': [2]}},
        ...     'output_features': {'action': {'shape': [2]}},
        ...     'horizon': 16
        ... }
        >>> policy = DiffusionPolicy(config)
        >>> obs = torch.randn(32, 2)  # batch_size=32, obs_dim=2
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
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
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
        
    def forward(self, obs: torch.Tensor, noisy_actions: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of diffusion policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            noisy_actions: Noisy action sequence [batch_size, action_dim]
            timestep: Diffusion timestep [batch_size] (optional, will use random if not provided)
            
        Returns:
            Predicted noise to subtract from noisy actions
        """
        batch_size = obs.shape[0]
        
        # Handle timestep
        if timestep is None:
            timestep = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=obs.device)
        
        # Ensure proper shapes
        if noisy_actions.dim() == 1:
            noisy_actions = noisy_actions.unsqueeze(0)
        if noisy_actions.shape[0] != batch_size:
            noisy_actions = noisy_actions.expand(batch_size, -1)
        
        # Encode inputs
        obs_emb = self.obs_encoder(obs)
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
    
    def sample_action(self, obs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample actions using DDPM sampling.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            num_samples: Number of action samples to generate
            
        Returns:
            Sampled actions [batch_size, action_dim]
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Start with random noise
        actions = torch.randn(batch_size, self.action_dim, device=device)
        
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
                
                # Add noise for non-final steps
                if t > 0:
                    noise = torch.randn_like(actions)
                    actions = actions + torch.sqrt(coeff1) * noise
            else:
                # Final step
                actions = (actions - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
        
        return actions
    
    def get_loss(self, trajectory_obs: torch.Tensor, trajectory_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute mean diffusion loss for a trajectory.
        
        Args:
            trajectory_obs: Observation tensor for a trajectory [num_steps, obs_dim]
            trajectory_actions: Ground truth actions for a trajectory [num_steps, action_dim]
            
        Returns:
            Mean diffusion loss over the trajectory
        """
        num_steps = trajectory_actions.shape[0]
        device = trajectory_actions.device
        
        # Sample random timesteps for each step in the trajectory
        timesteps = torch.randint(0, self.num_diffusion_steps, (num_steps,), device=device)
        
        # Sample noise
        noise = torch.randn_like(trajectory_actions)
        
        # Add noise to actions for all steps at once
        alpha_cumprod_t = self.alphas_cumprod[timesteps]
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t).unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t).unsqueeze(-1)
        
        noisy_actions = sqrt_alpha_cumprod_t * trajectory_actions + sqrt_one_minus_alpha_cumprod_t * noise
        
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
    
    def load_policy(self, filepath: Path) -> DiffusionPolicy:
        """Load policy from checkpoint."""
        filepath = Path(filepath)
        logger.info(f"Loading policy: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create policy with saved config
        config = checkpoint.get('config', self.load_policy_config())
        policy = DiffusionPolicy(config).to(self.device)
        
        # Load state dict
        policy.load_state_dict(checkpoint['model_state_dict'])
        
        return policy 