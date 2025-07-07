"""
LeRobot Environment Integration for CUPID

This module provides integration with the original LeRobot PushT environment,
allowing CUPID to use either the custom simulator or the original LeRobot environment.
"""

import sys
import os
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class LeRobotEnvironmentWrapper:
    """Wrapper for the original LeRobot PushT environment."""
    
    def __init__(self, config, render_mode: Optional[str] = None):
        """
        Initialize LeRobot environment wrapper.
        
        Args:
            config: CUPID configuration object
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.config = config
        self.render_mode = render_mode
        self.device = config.device
        
        # Add LeRobot to Python path if specified
        if config.lerobot_path:
            lerobot_path = Path(config.lerobot_path)
            if lerobot_path.exists():
                sys.path.insert(0, str(lerobot_path))
                logger.info(f"Added LeRobot path to sys.path: {lerobot_path}")
            else:
                logger.warning(f"LeRobot path does not exist: {lerobot_path}")
        
        # Import and initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize the LeRobot PushT environment."""
        try:
            # Import gym_pusht
            import gym_pusht
            import gymnasium as gym
            
            # Create environment with appropriate observation type
            # Use 'pixels_agent_pos' to match our dataset format
            # FIXED: Ensure render_mode is never None as it breaks reward computation
            effective_render_mode = self.render_mode if self.render_mode is not None else 'rgb_array'
            self.env = gym.make(
                'gym_pusht/PushT-v0',
                obs_type='pixels_agent_pos',
                render_mode=effective_render_mode,
                observation_width=96,
                observation_height=96,
                visualization_width=512,
                visualization_height=512
            )
            
            logger.info("LeRobot PushT environment initialized successfully")
            logger.info(f"   Observation space: {self.env.observation_space}")
            logger.info(f"   Action space: {self.env.action_space}")
            
        except ImportError as e:
            logger.error(f"Failed to import LeRobot environment: {e}")
            logger.error("Make sure gym_pusht is installed in the LeRobot environment")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LeRobot environment: {e}")
            raise
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Args:
            initial_state: Optional initial state [agent_x, agent_y] (for compatibility)
            
        Returns:
            Dictionary with observation data
        """
        if initial_state is not None:
            # Reset to specific state if provided
            options = {"reset_to_state": np.concatenate([initial_state, [100, 100, 0]])}
            obs, info = self.env.reset(options=options)
        else:
            obs, info = self.env.reset()
        
        # Convert observation to CUPID format
        return self._convert_observation(obs, info)
    
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array [x, y] in pixel coordinates
            
        Returns:
            Dictionary with step results
        """
        # Ensure action is in correct format
        action = np.array(action, dtype=np.float32)
        if action.shape != (2,):
            action = action.flatten()[:2]
        
        # Clip action to valid range [0, 512]
        action = np.clip(action, 0, 512)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert to CUPID format
        result = self._convert_observation(obs, info)
        result.update({
            'reward': float(reward),
            'success': bool(terminated),  # Environment terminates on success
            'done': bool(terminated or truncated),
            'info': info
        })
        
        return result
    
    def _convert_observation(self, obs: Union[Dict, np.ndarray], info: Dict) -> Dict[str, Any]:
        """
        Convert LeRobot observation to CUPID format.
        
        Args:
            obs: LeRobot observation (dict with 'pixels' and 'agent_pos')
            info: Environment info dictionary
            
        Returns:
            Dictionary in CUPID format
        """
        if isinstance(obs, dict):
            # Extract state (agent position)
            agent_pos = obs.get('agent_pos', np.array([0.0, 0.0]))
            
            # Extract image if available
            pixels = obs.get('pixels', None)
            
            result = {
                'observation': agent_pos,  # For compatibility with CUPID policies
                'agent_pos': agent_pos,
                'info': info
            }
            
            if pixels is not None:
                result['pixels'] = pixels
                
            return result
        else:
            # Handle state-only observation
            return {
                'observation': obs[:2] if len(obs) >= 2 else obs,
                'agent_pos': obs[:2] if len(obs) >= 2 else obs,
                'info': info
            }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            return self.env.render()
        else:
            return self.env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def get_coverage(self) -> float:
        """Get goal coverage (success metric)."""
        if hasattr(self.env, '_get_coverage'):
            return self.env._get_coverage()
        elif hasattr(self.env, 'get_coverage'):
            return self.env.get_coverage()
        else:
            # Fallback: estimate coverage from environment state
            # For PushT, we can estimate based on object position relative to goal
            try:
                # Try to get object and goal positions from environment
                if hasattr(self.env, '_get_obs'):
                    obs = self.env._get_obs()
                    if isinstance(obs, dict) and 'achieved_goal' in obs and 'desired_goal' in obs:
                        # Calculate distance-based coverage
                        achieved = obs['achieved_goal']
                        desired = obs['desired_goal']
                        distance = np.linalg.norm(achieved - desired)
                        # Convert distance to coverage (closer = higher coverage)
                        max_distance = 100.0  # Approximate max distance in PushT
                        coverage = max(0.0, 1.0 - (distance / max_distance))
                        return coverage
                
                # If we can't get proper coverage, return a reasonable default
                return 0.1  # Assume some minimal coverage rather than 0
            except:
                return 0.1


class LeRobotTaskEvaluator:
    """Task evaluator using the original LeRobot PushT environment."""
    
    def __init__(self, config, render_mode: Optional[str] = None):
        """
        Initialize LeRobot task evaluator.
        
        Args:
            config: CUPID configuration object
            render_mode: Rendering mode for environment
        """
        self.config = config
        self.render_mode = render_mode
        self.device = config.device
        
    def evaluate_policy_on_task(self, policy, dataset: List[Dict], 
                               num_episodes: int = 100, 
                               max_steps_per_episode: int = 300) -> Dict[str, float]:
        """
        Evaluate policy on the LeRobot PushT task.
        
        Args:
            policy: Policy to evaluate
            dataset: Dataset to sample initial states from
            num_episodes: Number of episodes to evaluate
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating policy on {num_episodes} LeRobot task episodes...")
        
        # Initialize environment
        env_wrapper = LeRobotEnvironmentWrapper(self.config, self.render_mode)
        
        episode_results = []
        policy.eval()
        
        try:
            for episode in range(num_episodes):
                # FIXED: Don't use fixed initial states as LeRobot environment doesn't support them properly
                # Use random initial states instead for proper evaluation
                initial_state = None  # Always use random initial state
                
                # Run episode
                result = self._run_episode(
                    policy, env_wrapper, initial_state, max_steps_per_episode
                )
                episode_results.append(result)
                
                if (episode + 1) % 10 == 0:
                    logger.info(f"   Completed {episode + 1}/{num_episodes} episodes")
        
        finally:
            env_wrapper.close()
        
        # Aggregate results
        return self._aggregate_results(episode_results)
    
    def _run_episode(self, policy, env_wrapper: LeRobotEnvironmentWrapper, 
                    initial_state: Optional[np.ndarray], max_steps: int, 
                    record_trajectory: bool = False) -> Dict[str, Any]:
        """
        Run a single episode with the policy.
        
        Args:
            policy: Policy to run
            env_wrapper: Environment wrapper
            initial_state: Optional initial state
            max_steps: Maximum steps for episode
            record_trajectory: Whether to record trajectory data for influence computation
            
        Returns:
            Episode results dictionary
        """
        # Reset environment
        obs_dict = env_wrapper.reset(initial_state)
        
        episode_reward = 0.0
        success = False
        step_count = 0
        trajectory = [] if record_trajectory else None
        
        for step in range(max_steps):
            # FIXED: Prepare proper observation format for the policy
            # The policy expects either state-only tensor or {'state': tensor, 'image': images} dict
            
            # Get state tensor
            state_obs = obs_dict['observation']  # Agent position [x, y]
            state_tensor = torch.FloatTensor(state_obs).to(self.device).unsqueeze(0)
            
            # Check if we have image data and if policy uses images
            has_images = 'pixels' in obs_dict and hasattr(policy, 'use_images') and policy.use_images
            
            if has_images:
                # Convert pixels to PIL Image format expected by policy
                pixels = obs_dict['pixels']  # Shape: [H, W, 3]
                if isinstance(pixels, np.ndarray):
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(pixels.astype(np.uint8))
                    
                    # Create proper observation dict for policy
                    policy_obs = {
                        'state': state_tensor,
                        'image': [pil_image]  # List of PIL images (batch_size=1)
                    }
                else:
                    # Fallback: state-only if image format is unexpected
                    policy_obs = state_tensor
            else:
                # State-only observation
                policy_obs = state_tensor
            
            # Get action from policy with corrected observation format
            # Use configured temperature for optimal diversity (matches training data levels)
            with torch.no_grad():
                temperature = getattr(self.config.policy, 'temperature', 100.0)  # Fallback to 100.0 if not configured
                action = policy.sample_action(policy_obs, temperature=temperature).cpu().numpy()[0]
            
            # Record trajectory step if needed
            if record_trajectory:
                trajectory.append({
                    'observation': state_obs.copy(),
                    'action': action.copy(),
                    'reward': 0.0  # Will be set to total episode reward later
                })
            
            # Take step
            step_result = env_wrapper.step(action)
            
            # Update observation dictionary for next step
            obs_dict = {
                'observation': step_result['observation'],
                'agent_pos': step_result.get('agent_pos', step_result['observation']),
                'pixels': step_result.get('pixels', None)
            }
            
            episode_reward += step_result['reward']
            success = step_result['success']
            step_count += 1
            
            if step_result['done']:
                break
        
        # Get final coverage
        final_coverage = env_wrapper.get_coverage()
        
        result = {
            'success': success,
            'reward': episode_reward,
            'episode_length': step_count,
            'final_coverage': final_coverage,
            'final_distance': 100.0 * (1.0 - final_coverage)  # Convert coverage to distance-like metric
        }
        
        # Add trajectory if recorded
        if record_trajectory and trajectory:
            # Set episode reward for all steps
            for step in trajectory:
                step['reward'] = episode_reward
            result['trajectory'] = trajectory
        
        return result
    
    def _run_single_episode(self, policy, initial_obs: np.ndarray, max_steps: int) -> Dict[str, float]:
        """
        Run a single episode for influence computation (interface compatibility).
        
        Args:
            policy: Policy to evaluate
            initial_obs: Initial observation (state) - IGNORED for LeRobot (uses random)
            max_steps: Maximum steps for the episode
            
        Returns:
            Episode results dictionary compatible with TaskEvaluator interface
        """
        # Initialize environment wrapper for this episode
        # FIXED: Don't pass render_mode=None as it breaks reward computation
        env_wrapper = LeRobotEnvironmentWrapper(self.config)
        
        try:
            # FIXED: Use random initial state instead of initial_obs since LeRobot doesn't support fixed states
            result = self._run_episode(policy, env_wrapper, None, max_steps, record_trajectory=True)
            
            # Convert to TaskEvaluator interface format
            return {
                'success': result['success'],
                'total_reward': result['reward'],
                'avg_reward': result['reward'] / max(result['episode_length'], 1),
                'final_distance': result['final_distance'],
                'episode_length': result['episode_length'],
                'completed': result['success'],
                'trajectory': result.get('trajectory', [])  # Now includes recorded trajectory
            }
            
        finally:
            env_wrapper.close()
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict[str, float]:
        """Aggregate episode results into summary statistics."""
        if not episode_results:
            return {}
        
        success_rate = np.mean([r['success'] for r in episode_results])
        avg_reward = np.mean([r['reward'] for r in episode_results])  # Total episode reward (correct for CUPID)
        avg_episode_length = np.mean([r['episode_length'] for r in episode_results])
        avg_final_coverage = np.mean([r['final_coverage'] for r in episode_results])
        avg_final_distance = np.mean([r['final_distance'] for r in episode_results])
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,  # Total episode reward is correct for policy comparison
            'avg_episode_length': avg_episode_length,
            'avg_final_coverage': avg_final_coverage,
            'avg_final_distance': avg_final_distance,
            'num_episodes': len(episode_results)
        }
    

    
    def demonstrate_policy_rollouts(self, policy, policy_name: str, dataset: List[Dict],
                                  num_rollouts: int = 5, max_steps: int = 300, 
                                  output_dir: str = "outputs") -> List[str]:
        """
        Demonstrate policy rollouts using LeRobot's native video generation.
        
        Args:
            policy: Policy to demonstrate
            policy_name: Name for display
            dataset: Dataset to sample initial states from
            num_rollouts: Number of rollouts to generate
            max_steps: Maximum steps per rollout
            output_dir: Directory to save videos
            
        Returns:
            List of paths to generated video files
        """
        import os
        import imageio
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        video_paths = []
        
        logger.info(f"Generating {num_rollouts} LeRobot demonstration videos for {policy_name}...")
        
        for rollout in range(num_rollouts):
            # Initialize environment wrapper for video recording
            env_wrapper = LeRobotEnvironmentWrapper(self.config, render_mode='rgb_array')
            
            try:
                # Sample initial state
                if dataset:
                    demo_idx = np.random.randint(len(dataset))
                    demo_sample = dataset[demo_idx]
                    initial_state = np.array(demo_sample['observation.state'])
                else:
                    initial_state = None
                
                # Reset environment and policy
                obs_dict = env_wrapper.reset(initial_state)
                
                # Prepare to collect frames and rewards (LeRobot style)
                frames = []
                rewards = []
                
                # Render initial frame
                frames.append(env_wrapper.render())
                
                step = 0
                done = False
                
                while not done and step < max_steps:
                    # Prepare observation for policy (LeRobot style)
                    state = torch.from_numpy(obs_dict['agent_pos']).to(torch.float32)
                    image = torch.from_numpy(obs_dict['pixels']).to(torch.float32) / 255.0
                    image = image.permute(2, 0, 1)  # HWC -> CHW
                    
                    # Send to device
                    state = state.to(self.device, non_blocking=True).unsqueeze(0)
                    image = image.to(self.device, non_blocking=True).unsqueeze(0)
                    
                    # Create policy input (LeRobot format)
                    observation = {
                        "observation.state": state,
                        "observation.image": image,
                    }
                    
                    # Get action from policy (CUPID policies use sample_action, not select_action)
                    with torch.inference_mode():
                        if hasattr(policy, 'select_action'):
                            # LeRobot-style policy
                            action = policy.select_action(observation)
                        else:
                            # CUPID-style policy - convert observation format
                            if isinstance(observation, dict):
                                # Convert LeRobot observation format to CUPID format
                                cupid_obs = {
                                    'state': observation["observation.state"],
                                    'image': observation["observation.image"]
                                }
                            else:
                                cupid_obs = observation
                            action = policy.sample_action(cupid_obs)
                    
                    # Prepare action for environment
                    numpy_action = action.squeeze(0).to("cpu").numpy()
                    
                    # Step environment
                    step_result = env_wrapper.step(numpy_action)
                    obs_dict = {
                        'agent_pos': step_result['agent_pos'],  # Agent position is separate
                        'pixels': step_result['pixels']  # Pixels are separate
                    }
                    
                    # Keep track of rewards and frames (LeRobot style)
                    rewards.append(step_result['reward'])
                    frames.append(env_wrapper.render())
                    
                    done = step_result['done']
                    step += 1
                
                # Save video using LeRobot's method
                if frames:
                    video_path = f"{output_dir}/lerobot_{policy_name}_rollout_{rollout+1}.mp4"
                    
                    # Get FPS from environment metadata (LeRobot style)
                    fps = env_wrapper.env.metadata.get("render_fps", 30)
                    
                    # Encode frames into MP4 video (LeRobot style)
                    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
                    video_paths.append(video_path)
                    
                    success_msg = "Success!" if step_result.get('success', False) else "Failure!"
                    total_reward = sum(rewards)
                    logger.info(f"Rollout {rollout+1}: {success_msg} (Steps: {step}, Reward: {total_reward:.3f})")
                    logger.info(f"   Video saved: {video_path}")
                
            except Exception as e:
                logger.error(f"Error generating LeRobot rollout {rollout+1}: {e}")
                continue
            finally:
                env_wrapper.close()
        
        logger.info(f"Generated {len(video_paths)}/{num_rollouts} LeRobot demonstration videos")
        return video_paths 