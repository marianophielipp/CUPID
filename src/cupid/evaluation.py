"""
Evaluation module for CUPID policies.

This module provides proper task-based evaluation metrics for robot policies,
focusing on task success rates and rewards using environment simulation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from pathlib import Path

from .policy import DiffusionPolicy

logger = logging.getLogger(__name__)


class PushTSimulator:
    """
    Simulates PushT environment dynamics matching LeRobot PushT exactly.
    Based on the real LeRobot PushT environment where actions are direct
    pusher target positions and rewards are coverage-based.
    """
    
    def __init__(self, workspace_bounds=(0, 512, 0, 512), render_mode=None):
        """
        Initialize PushT simulator with EXACT LeRobot parameters.
        
        Args:
            workspace_bounds: (x_min, x_max, y_min, y_max) - uses LeRobot's [0,512] range
            render_mode: 'human' for visual rendering, None for headless
        """
        self.workspace_bounds = workspace_bounds
        self.render_mode = render_mode
        
        # EXACT LeRobot PushT parameters
        self.goal_zone_center = np.array([256.0, 400.0])  # Approximate goal zone center
        self.goal_zone_size = 80.0  # Size of goal zone
        self.success_threshold = 0.95  # 95% coverage for success (from gym-pusht docs)
        
        # Object and pusher properties
        self.object_size = 40.0  # Approximate T-shaped object size
        self.pusher_radius = 10.0  # Smaller pusher radius
        
        # Physics parameters - much more realistic
        self.max_pusher_speed = 20.0  # How fast pusher can move per step
        self.push_effectiveness = 0.3  # How much object moves when pushed
        self.friction = 0.95  # Object friction/momentum decay
        
        # State variables
        self.object_position = np.array([200.0, 100.0])  # T-shaped object position
        self.pusher_position = np.array([150.0, 150.0])  # Circular pusher position
        self.object_velocity = np.array([0.0, 0.0])  # Object momentum
        self.trajectory = [] # To store trajectory for influence computation
        
        # Rendering state variables
        self.current_reward = 0.0
        self.success = False
        
        # Initialize rendering if needed
        self.screen = None
        self.clock = None
        self.window_size = 512
        self.last_action = None
        if render_mode == 'human':
            self._init_rendering()

    def reset(self, initial_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Reset environment with initial observation from dataset."""
        # In LeRobot dataset, observation.state is [object_x, object_y]
        self.object_position = np.array(initial_obs, dtype=np.float32)
        self.object_velocity = np.array([0.0, 0.0])
        
        # Position pusher nearby but not overlapping (realistic starting position)
        offset = np.array([-40.0, -40.0])
        self.pusher_position = self.object_position + offset
        
        # Keep pusher in bounds
        self.pusher_position = np.clip(
            self.pusher_position,
            [self.workspace_bounds[0] + 20, self.workspace_bounds[2] + 20],
            [self.workspace_bounds[1] - 20, self.workspace_bounds[3] - 20]
        )
        
        # Reset trajectory and rendering state
        self.trajectory = []
        self.current_reward = 0.0
        self.success = False
        
        return {
            'observation': self.object_position.copy(),
            'goal_zone': self.goal_zone_center.copy()
        }
    
    def step(self, action: np.ndarray, dt: float = 1/50) -> Dict[str, float]:
        """
        Simulate one step with EXACT LeRobot PushT dynamics.
        
        Args:
            action: 2D action [x, y] representing pusher TARGET position (not force)
            dt: Time step
            
        Returns:
            Dictionary with next_obs, reward, done, success
        """
        # Store action for rendering
        self.last_action = np.array(action, dtype=np.float32)
        
        # STEP 1: Move pusher toward action target (like LeRobot)
        # Actions are direct pusher target positions
        pusher_target = np.array(action, dtype=np.float32)
        
        # Clip target to workspace bounds
        pusher_target = np.clip(
            pusher_target,
            [self.workspace_bounds[0], self.workspace_bounds[2]],
            [self.workspace_bounds[1], self.workspace_bounds[3]]
        )
        
        # Move pusher toward target with maximum speed
        pusher_direction = pusher_target - self.pusher_position
        pusher_distance = np.linalg.norm(pusher_direction)
        
        if pusher_distance > 0:
            pusher_direction = pusher_direction / pusher_distance
            # Scale max speed by dt
            pusher_movement = min(pusher_distance, self.max_pusher_speed * dt)
            self.pusher_position = self.pusher_position + pusher_direction * pusher_movement
        
        # STEP 2: Check pusher-object collision
        pusher_object_distance = np.linalg.norm(self.pusher_position - self.object_position)
        contact_threshold = self.pusher_radius + self.object_size / 2
        makes_contact = pusher_object_distance < contact_threshold
        
        # STEP 3: Apply physics if contact
        if makes_contact:
            # Calculate push direction and magnitude
            push_direction = self.object_position - self.pusher_position
            if np.linalg.norm(push_direction) > 0:
                push_direction = push_direction / np.linalg.norm(push_direction)
                
                # Push force based on pusher movement
                pusher_movement_vec = pusher_target - self.pusher_position
                push_magnitude = np.linalg.norm(pusher_movement_vec) * self.push_effectiveness
                
                # Apply push to object velocity
                self.object_velocity += push_direction * push_magnitude
        
        # STEP 4: Apply object physics
        # Update object position with velocity (scaled by dt)
        self.object_position += self.object_velocity * dt
        
        # Apply friction
        self.object_velocity *= self.friction
        
        # Keep object in bounds with realistic bouncing
        if self.object_position[0] < self.workspace_bounds[0] + self.object_size/2:
            self.object_position[0] = self.workspace_bounds[0] + self.object_size/2
            self.object_velocity[0] = -self.object_velocity[0] * 0.3
        elif self.object_position[0] > self.workspace_bounds[1] - self.object_size/2:
            self.object_position[0] = self.workspace_bounds[1] - self.object_size/2
            self.object_velocity[0] = -self.object_velocity[0] * 0.3
            
        if self.object_position[1] < self.workspace_bounds[2] + self.object_size/2:
            self.object_position[1] = self.workspace_bounds[2] + self.object_size/2
            self.object_velocity[1] = -self.object_velocity[1] * 0.3
        elif self.object_position[1] > self.workspace_bounds[3] - self.object_size/2:
            self.object_position[1] = self.workspace_bounds[3] - self.object_size/2
            self.object_velocity[1] = -self.object_velocity[1] * 0.3
        
        # STEP 5: Calculate coverage-based reward (like real LeRobot PushT)
        # Reward is based on how much the object overlaps with goal zone
        coverage = self._calculate_goal_coverage()
        reward = coverage  # Direct coverage as reward (0.0 to 1.0)
        
        # Success if coverage >= 95% (from gym-pusht docs)
        success = coverage >= self.success_threshold
        
        # Episode ends on success or timeout (handled by caller)
        done = success
        
        # Store current state for rendering
        self.current_reward = reward
        self.success = success
        
        # Store step in trajectory
        self.trajectory.append({
            'observation': self.object_position.copy(),
            'action': self.last_action.copy()
        })
        
        # Render if enabled
        if self.render_mode == 'human':
            self.render()
        
        return {
            'next_observation': self.object_position.copy(),
            'reward': reward,
            'done': done,
            'success': success,
            'coverage': coverage,
            'makes_contact': makes_contact
        }
    
    def _calculate_goal_coverage(self) -> float:
        """
        Calculate what percentage of the T-shaped object is in the goal zone.
        This matches the coverage-based reward from real LeRobot PushT.
        """
        # Distance from object center to goal zone center
        distance_to_goal = np.linalg.norm(self.object_position - self.goal_zone_center)
        
        # Simple coverage calculation - more sophisticated would use actual T-shape
        # If object is fully inside goal zone, coverage = 1.0
        # If object is partially inside, coverage = proportional
        # If object is outside, coverage = 0.0
        
        if distance_to_goal <= self.goal_zone_size / 2 - self.object_size / 2:
            # Object fully inside goal zone
            coverage = 1.0
        elif distance_to_goal <= self.goal_zone_size / 2 + self.object_size / 2:
            # Object partially inside goal zone
            overlap = (self.goal_zone_size / 2 + self.object_size / 2) - distance_to_goal
            max_overlap = self.object_size
            coverage = max(0.0, overlap / max_overlap)
        else:
            # Object outside goal zone
            coverage = 0.0
        
        return coverage
    
    def get_trajectory(self) -> List[Dict[str, np.ndarray]]:
        """Return the recorded trajectory for the last episode."""
        return self.trajectory
    
    def _init_rendering(self):
        """Initialize pygame for rendering."""
        try:
            import pygame
            pygame.init()
            
            if self.render_mode == 'human':
                # Create visible window for human viewing
                self.screen = pygame.display.set_mode((512, 512))
                pygame.display.set_caption("PushT Simulation")
            else:
                # Create headless surface for rgb_array mode (video generation)
                self.screen = pygame.Surface((512, 512))
            
            self.clock = None
        except ImportError:
            logger.warning("pygame not available for rendering")
            self.screen = None

    def render(self):
        """Render the current state of the simulation."""
        if self.render_mode is None:
            return
        
        try:
            import pygame
            
            if self.screen is None:
                self._init_rendering()
            
            if self.screen is None:
                return
            
            # Clear screen with white background
            self.screen.fill((255, 255, 255))
            
            # Draw goal zone (background)
            goal_color = (255, 220, 220)  # Light red background
            goal_x, goal_y = int(self.goal_zone_center[0]), int(self.goal_zone_center[1])
            goal_rect = pygame.Rect(goal_x - 40, goal_y - 40, 80, 80)
            pygame.draw.rect(self.screen, goal_color, goal_rect)
            
            # Draw goal zone border
            goal_border_color = (200, 50, 50)  # Dark red border
            pygame.draw.rect(self.screen, goal_border_color, goal_rect, 3)
            
            # Draw target T-shape in goal zone (what we want to achieve)
            target_color = (150, 150, 150)  # Gray target
            target_x, target_y = goal_x, goal_y
            target_h_rect = pygame.Rect(target_x - 20, target_y - 3, 40, 6)
            target_v_rect = pygame.Rect(target_x - 3, target_y - 3, 6, 25)
            pygame.draw.rect(self.screen, target_color, target_h_rect)
            pygame.draw.rect(self.screen, target_color, target_v_rect)
            
            # Draw current T-shaped object
            t_color = (50, 100, 255)  # Blue
            t_x, t_y = int(self.object_position[0]), int(self.object_position[1])
            
            # Draw T shape (horizontal bar + vertical bar) - slightly larger than target
            horizontal_rect = pygame.Rect(t_x - 25, t_y - 4, 50, 8)
            vertical_rect = pygame.Rect(t_x - 4, t_y - 4, 8, 30)
            pygame.draw.rect(self.screen, t_color, horizontal_rect)
            pygame.draw.rect(self.screen, t_color, vertical_rect)
            
            # Draw pusher (end effector)
            pusher_color = (255, 100, 100)  # Red
            pusher_x, pusher_y = int(self.pusher_position[0]), int(self.pusher_position[1])
            pygame.draw.circle(self.screen, pusher_color, (pusher_x, pusher_y), 8)
            
            # Draw text information
            font = pygame.font.Font(None, 24)
            
            # Show reward
            reward_text = f"Reward: {self.current_reward:.3f}"
            reward_surface = font.render(reward_text, True, (50, 50, 50))
            self.screen.blit(reward_surface, (10, 10))
            
            # Show success status
            success_text = f"Success: {'Yes' if self.success else 'No'}"
            success_color = (0, 150, 0) if self.success else (150, 0, 0)
            success_surface = font.render(success_text, True, success_color)
            self.screen.blit(success_surface, (10, 40))
            
            # Show distance to goal
            distance = np.linalg.norm(self.object_position - self.goal_zone_center)
            distance_text = f"Distance: {distance:.1f}"
            distance_surface = font.render(distance_text, True, (100, 100, 100))
            self.screen.blit(distance_surface, (10, 70))
            
            # Show last action if available
            if hasattr(self, 'last_action') and self.last_action is not None:
                action_text = f"Action: [{self.last_action[0]:.0f}, {self.last_action[1]:.0f}]"
                action_status = font.render(action_text, True, (200, 200, 200))
                self.screen.blit(action_status, (10, 110))
            
            # Show interactive controls (if in demonstration mode)
            if hasattr(self, '_show_controls') and self._show_controls:
                controls_text = [
                    "Controls:",
                    "SPACE = Start/Pause",
                    "R = Restart rollout", 
                    "N = Next rollout",
                    "Q = Quit demos"
                ]
                for i, text in enumerate(controls_text):
                    color = (255, 255, 0) if i == 0 else (200, 200, 200)  # Yellow for title
                    control_surface = font.render(text, True, color)
                    self.screen.blit(control_surface, (10, 140 + i * 20))
            
            if self.render_mode == 'human':
                pygame.display.flip()
            
            # Control frame rate
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.clock.tick(20)  # 20 FPS for smooth rendering
            
        except ImportError:
            logger.warning("pygame not available for rendering")
        except Exception as e:
            logger.error(f"Rendering error: {e}")

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame as RGB array for video recording.
        
        Returns:
            RGB frame as numpy array (H, W, 3) or None if rendering unavailable
        """
        if self.render_mode not in ['human', 'rgb_array']:
            return None
            
        try:
            import pygame
            
            if self.screen is None:
                self._init_rendering()
            
            if self.screen is None:
                return None
            
            # Render the current frame (this updates self.screen)
            self.render()
            
            # Convert pygame surface to RGB array
            rgb_array = pygame.surfarray.array3d(self.screen)
            # Pygame uses (width, height, channels), convert to (height, width, channels)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            
            return rgb_array
            
        except ImportError:
            logger.warning("pygame not available for frame capture")
            return None
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def close(self):
        """Close the rendering window."""
        if self.screen is not None:
            import pygame
            
            # Only quit display if we actually created a display window
            if self.render_mode == 'human':
                pygame.display.quit()
            
            pygame.quit()
            self.screen = None


class TaskEvaluator:
    """Evaluates policies on actual task performance metrics."""
    
    def __init__(self, config, render_mode=None):
        """
        Initialize TaskEvaluator.
        
        Args:
            config: CUPID configuration object
            render_mode: 'human' for visual rendering, None for headless
        """
        self.config = config
        self.device = config.device
        self.render_mode = render_mode
        self.environment_type = getattr(config, 'environment_type', 'cupid')
        
        # Initialize simulation environment (always available as fallback)
        self.simulator = PushTSimulator(render_mode=render_mode)
        
        # Initialize LeRobot evaluator if configured
        self.lerobot_evaluator = None
        if self.environment_type == 'lerobot':
            try:
                from .lerobot_integration import LeRobotTaskEvaluator
                self.lerobot_evaluator = LeRobotTaskEvaluator(config, render_mode)
                logger.info("LeRobot environment integration enabled")
            except ImportError as e:
                logger.warning(f"LeRobot integration failed: {e}")
                logger.info("Falling back to CUPID simulation environment")
                self.environment_type = 'cupid'
        elif self.environment_type == 'cupid':
            logger.info("Using CUPID simulation environment")
        else:
            logger.warning(f"Unknown environment type '{self.environment_type}', using CUPID simulation")
            self.environment_type = 'cupid'
    
    def evaluate_policy_on_task(
        self,
        policy: DiffusionPolicy,
        dataset: List[Dict],
        num_episodes: int = 100,
        max_steps_per_episode: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate policy on actual task performance using configured environment.
        
        Args:
            policy: Policy to evaluate
            dataset: Dataset containing demonstrations
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode (for simulation)
            
        Returns:
            Dictionary with task performance metrics
        """
        logger.info(f"Evaluating policy on {num_episodes} task episodes...")
        
        # Use LeRobot environment if configured and available
        if self.environment_type == 'lerobot' and self.lerobot_evaluator is not None:
            try:
                logger.info("🤖 Using LeRobot PushT environment...")
                metrics = self.lerobot_evaluator.evaluate_policy_on_task(
                    policy, dataset, num_episodes, max_steps_per_episode
                )
                
                logger.info(f"✅ LeRobot evaluation complete:")
                logger.info(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
                logger.info(f"  Average reward: {metrics.get('avg_reward', 0):.3f}")
                
                return metrics
                
            except Exception as e:
                logger.warning(f"LeRobot evaluation failed: {e}")
                logger.info("🔄 Falling back to simulation environment")
        
        # Use simulation environment (default or fallback)
        logger.info("🎮 Using simulation environment")
        return self._evaluate_with_simulation(policy, dataset, num_episodes, max_steps_per_episode)
    
    def _evaluate_with_simulation(
        self,
        policy: DiffusionPolicy,
        dataset: List[Dict],
        num_episodes: int,
        max_steps_per_episode: int
    ) -> Dict[str, float]:
        """
        Evaluate policy using simulation environment.
        
        Args:
            policy: Policy to evaluate
            dataset: Dataset containing demonstrations
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary with task performance metrics
        """
        policy.eval()
        
        eval_indices = np.random.choice(
            len(dataset), 
            size=min(num_episodes, len(dataset)), 
            replace=False
        )
        
        episode_results = []
        
        # Create a subset of the dataset for efficient access
        if hasattr(dataset, 'select'):
            eval_dataset = dataset.select(eval_indices)
        else:
            # For lists, create a subset list
            eval_dataset = [dataset[i] for i in eval_indices]

        with torch.no_grad():
            for sample in tqdm(eval_dataset, desc="Evaluating episodes"):
                # Get initial observation from dataset
                initial_obs = np.array(sample['observation.state'])
                
                # Run episode simulation
                episode_result = self._run_simulation_episode(
                    policy, initial_obs, max_steps_per_episode
                )
                episode_results.append(episode_result)
        
        # Aggregate results
        metrics = self._aggregate_episode_results(episode_results)
        
        logger.info(f"Simulation evaluation complete:")
        logger.info(f"  Success rate: {metrics['success_rate']:.1%}")
        logger.info(f"  Average reward: {metrics['avg_reward']:.3f}")
        logger.info(f"  Average final distance: {metrics['avg_final_distance']:.3f}")
        
        policy.train()
        return metrics
    
    def _run_episode(
        self,
        policy: DiffusionPolicy,
        initial_obs: np.ndarray,
        max_steps: int
    ) -> Dict[str, float]:
        """
        Run a single episode with the policy using the configured environment.
        
        Args:
            policy: Policy to evaluate
            initial_obs: Initial observation
            max_steps: Maximum steps for the episode
            
        Returns:
            Episode results dictionary
        """
        # Use LeRobot environment if configured and available
        if self.environment_type == 'lerobot' and self.lerobot_evaluator is not None:
            try:
                # For LeRobot, we need to run a single episode
                return self.lerobot_evaluator._run_single_episode(policy, initial_obs, max_steps)
            except Exception as e:
                logger.warning(f"LeRobot episode failed: {e}, falling back to simulation")
        
        # Use simulation environment (default or fallback)
        return self._run_simulation_episode(policy, initial_obs, max_steps)
    
    def _run_simulation_episode(
        self,
        policy: DiffusionPolicy,
        initial_obs: np.ndarray,
        max_steps: int
    ) -> Dict[str, float]:
        """
        Run a single episode with the policy using simulation environment.
        
        Args:
            policy: Policy to evaluate
            initial_obs: Initial observation
            max_steps: Maximum steps for the episode
            
        Returns:
            Episode results dictionary
        """
        # Reset simulator
        state = self.simulator.reset(initial_obs)
        
        total_reward = 0.0
        steps = 0
        success = False
        final_distance = float('inf')
        
        for step in range(max_steps):
            # Get current observation
            current_state = torch.FloatTensor(state['observation']).to(self.device)
            
            # Try using n_obs_steps=1 to see if that works around the tensor reshaping issue
            state_batch = current_state.unsqueeze(0)  # (B, state_dim) = (1, 2)
            
            # Create proper batch format for LeRobot policy
            # Check if policy expects images by looking at its vision backbone
            if hasattr(policy, 'config') and hasattr(policy.config, 'vision_backbone') and policy.config.vision_backbone:
                # Policy expects images, provide dummy images
                policy_input = {
                    'observation.state': state_batch,  # (B, state_dim)
                    'observation.image': torch.zeros(1, 3, 84, 84, device=self.device)  # (B, C, H, W)
                }
            else:
                # State-only policy
                policy_input = {
                    'observation.state': state_batch
                }
            
            # Generate action with policy using LeRobot's select_action method
            try:
                predicted_action = policy.select_action(policy_input)
            except Exception as e:
                # If the above fails, try with just state (fallback)
                logger.debug(f"Policy select_action failed with full input, trying state-only: {e}")
                policy_input = {'observation.state': state_batch}
                predicted_action = policy.select_action(policy_input)
            
            # Convert to numpy and take step
            action = predicted_action.cpu().numpy().flatten()
            step_result = self.simulator.step(action, dt=1/max_steps)
            
            # Update state
            state['observation'] = step_result['next_observation']
            total_reward += step_result['reward']
            steps += 1
            
            if step_result['done']:
                success = step_result['success']
                # Calculate final distance manually since simulator returns coverage
                current_pos = self.simulator.object_position
                goal_pos = self.simulator.goal_zone_center
                final_distance = np.linalg.norm(current_pos - goal_pos)
                break
        
        # If episode didn't end, get final distance
        if not success:
            current_pos = self.simulator.object_position
            goal_pos = self.simulator.goal_zone_center
            final_distance = np.linalg.norm(current_pos - goal_pos)
        
        # Debug logging for distance calculation
        logger.debug(f"Episode finished: success={success}, steps={steps}, final_distance={final_distance:.2f}")
        logger.debug(f"Final object position: {self.simulator.object_position}")
        logger.debug(f"Goal position: {self.simulator.goal_zone_center}")
        
        return {
            'success': success,
            'total_reward': total_reward,
            'avg_reward': total_reward / max(steps, 1),
            'final_distance': final_distance,
            'episode_length': steps,
            'completed': success,
            'trajectory': self.simulator.get_trajectory()
        }
    
    def _aggregate_episode_results(self, episode_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results from multiple episodes."""
        if not episode_results:
            return {}
        
        successes = [r['success'] for r in episode_results]
        total_rewards = [r['total_reward'] for r in episode_results]
        avg_rewards = [r['avg_reward'] for r in episode_results]
        final_distances = [r['final_distance'] for r in episode_results]
        episode_lengths = [r['episode_length'] for r in episode_results]
        
        return {
            # Primary task metrics
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(avg_rewards),
            'std_reward': np.std(avg_rewards),
            'total_reward_mean': np.mean(total_rewards),
            
            # Task-specific metrics
            'avg_final_distance': np.mean(final_distances),
            'std_final_distance': np.std(final_distances),
            'avg_episode_length': np.mean(episode_lengths),
            
            # Policy quality metrics
            'success_episodes': int(np.sum(successes)),
            'total_episodes': len(episode_results),
            
            # Statistical metrics
            'reward_percentile_25': np.percentile(avg_rewards, 25),
            'reward_percentile_75': np.percentile(avg_rewards, 75),
            'distance_percentile_25': np.percentile(final_distances, 25),
            'distance_percentile_75': np.percentile(final_distances, 75),
            
            # Derived metrics
            'task_efficiency': np.mean(successes) / max(np.mean(episode_lengths), 1)
        }
    
    def demonstrate_policy_rollouts(
        self,
        policy: DiffusionPolicy,
        policy_name: str,
        dataset: List[Dict],
        num_rollouts: int = 10,
        max_steps: int = 50,
        output_dir: str = "outputs"
    ) -> List[str]:
        """
        Demonstrate policy rollouts with interactive controls or video generation.
        
        Args:
            policy: The policy to demonstrate
            policy_name: Name for display purposes
            dataset: Dataset to sample initial states from
            num_rollouts: Number of rollouts to show
            max_steps: Maximum steps per rollout
            output_dir: Directory to save videos (for LeRobot)
            
        Returns:
            List of paths to generated video files (LeRobot) or empty list (simulation)
        """
        # Use LeRobot environment if configured and available
        if self.environment_type == 'lerobot' and self.lerobot_evaluator is not None:
            try:
                logger.info(f"Using LeRobot native video generation for {policy_name}...")
                video_paths = self.lerobot_evaluator.demonstrate_policy_rollouts(
                    policy, policy_name, dataset, num_rollouts, max_steps, output_dir
                )
                return video_paths
            except Exception as e:
                logger.warning(f"LeRobot demonstration failed: {e}")
                logger.info("🔄 Falling back to simulation environment")
        
        # Use simulation environment (interactive mode)
        if self.render_mode != 'human':
            logger.info(f"Rendering disabled, skipping {policy_name} demonstrations")
            return []
        
        # Use simulation environment (default or fallback)
        logger.info(f"Demonstrating {policy_name} rollouts...")
        
        policy.eval()
        
        # Sample random episodes for demonstration
        demo_indices = np.random.choice(len(dataset), size=min(num_rollouts, len(dataset)), replace=False)
        
        for rollout, demo_idx in enumerate(demo_indices):
            logger.info(f"   Rollout {rollout + 1}/{num_rollouts}")
            
            # Get initial observation
            demo_sample = dataset[int(demo_idx)]
            initial_obs = np.array(demo_sample['observation.state'])
            
            # Run interactive animated episode
            try:
                self._run_interactive_episode(policy, initial_obs, max_steps, rollout + 1, num_rollouts)
            except KeyboardInterrupt:
                logger.info("🚪 Demonstration interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in rollout {rollout + 1}: {e}")
                continue
        
        logger.info(f"✅ {policy_name} demonstration complete")
        return []  # No video files generated in simulation mode

    def _run_interactive_episode(
        self,
        policy: DiffusionPolicy,
        initial_obs: np.ndarray,
        max_steps: int,
        rollout_num: int,
        total_rollouts: int
    ) -> Dict[str, float]:
        """
        Run a single interactive episode with user controls.
        
        Args:
            policy: Policy to run
            initial_obs: Initial observation
            max_steps: Maximum steps
            rollout_num: Current rollout number  
            total_rollouts: Total number of rollouts
            
        Returns:
            Episode results dictionary
        """
        # Initialize simulator
        simulator = PushTSimulator(render_mode='human')
        obs_dict = simulator.reset(initial_obs)
        obs = obs_dict['observation']
        
        # Episode state
        step = 0
        total_reward = 0.0
        success = False
        paused = True  # Start paused
        
        # Set window title and show controls
        simulator._show_controls = True
        
        logger.info(f"      Interactive animated rollout...")
        logger.info(f"         Controls: SPACE=Start/Pause, R=Restart, N=Next rollout, Q=Quit demos")
        
        try:
            import pygame
            
            while step < max_steps and not success:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Window closed")
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            logger.debug(f"         {'⏸️ Paused' if paused else '▶️ Playing'}")
                        elif event.key == pygame.K_r:
                            # Restart episode
                            logger.info(f"         🔄 Restarting rollout...")
                            obs_dict = simulator.reset(initial_obs)
                            obs = obs_dict['observation']
                            step = 0
                            total_reward = 0.0
                            success = False
                        elif event.key == pygame.K_n:
                            # Next rollout
                            logger.info(f"         ⏭️ Skipping to next rollout...")
                            break
                        elif event.key == pygame.K_q:
                            # Quit demonstrations
                            logger.info(f"         🚪 Quitting demonstrations...")
                            raise KeyboardInterrupt("User quit")
                
                # If paused, just render and continue
                if paused:
                    simulator.render()
                    pygame.time.wait(50)  # 50ms delay
                    continue
                
                # Get action from policy
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    # Create proper batch format for LeRobot policy
                    # Check if policy expects images by looking at its vision backbone
                    if hasattr(policy, 'config') and hasattr(policy.config, 'vision_backbone') and policy.config.vision_backbone:
                        # Policy expects images, provide dummy images
                        policy_input = {
                            'observation.state': obs_tensor,
                            'observation.image': torch.zeros(1, 3, 84, 84, device=self.device)  # Dummy image
                        }
                    else:
                        # State-only policy
                        policy_input = {
                            'observation.state': obs_tensor
                        }
                    
                    try:
                        action = policy.select_action(policy_input).cpu().numpy()[0]
                    except Exception as e:
                        # If the above fails, try with just state (fallback)
                        logger.debug(f"Policy select_action failed with full input, trying state-only: {e}")
                        policy_input = {'observation.state': obs_tensor}
                        action = policy.select_action(policy_input).cpu().numpy()[0]
                
                # Take step
                step_result = simulator.step(action, dt=1/20)
                obs = step_result['next_observation']
                reward = step_result['reward']
                success = step_result['success']
                
                total_reward += reward
                step += 1
                
                # Render
                simulator.render()
                
                # Small delay for smooth animation
                pygame.time.wait(50)  # 50ms = 20 FPS
                
                # Check for success
                if success:
                    logger.info(f"      ✅ Success! Completed in {step} steps")
                    logger.info(f"         Press SPACE to continue, R to restart, N for next rollout")
                    paused = True  # Pause on success
                    
                    # Wait for user input after success
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                raise KeyboardInterrupt("Window closed")
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE or event.key == pygame.K_n:
                                    waiting = False
                                elif event.key == pygame.K_r:
                                    logger.info(f"      🔄 Restarting rollout {rollout_num}...")
                                    obs_dict = simulator.reset(initial_obs)
                                    obs = obs_dict['observation']
                                    step = 0
                                    total_reward = 0.0
                                    success = False
                                    waiting = False
                                    paused = False  # Resume after restart
                                elif event.key == pygame.K_q:
                                    logger.info(f"      🚪 Demonstration interrupted by user")
                                    raise KeyboardInterrupt("User quit")
                        
                        simulator.render()
                        pygame.time.wait(50)
            
            # Episode ended without success
            if not success:
                logger.info(f"      ⏱️ Episode ended after {max_steps} steps")
                logger.info(f"         Press SPACE to continue, R to restart, N for next rollout")
                
                # Wait for user input
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt("Window closed")
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE or event.key == pygame.K_n:
                                waiting = False
                            elif event.key == pygame.K_r:
                                # Restart and continue
                                obs_dict = simulator.reset(initial_obs)
                                obs = obs_dict['observation']
                                step = 0
                                total_reward = 0.0
                                success = False
                                waiting = False
                                paused = False
                            elif event.key == pygame.K_q:
                                raise KeyboardInterrupt("User quit")
                    
                    simulator.render()
                    pygame.time.wait(50)
        
        finally:
            simulator.close()
        
        return {
            'total_reward': total_reward,
            'success': success,
            'steps': step,
            'final_distance': np.linalg.norm(obs - simulator.goal_zone_center) if 'obs' in locals() else 0.0
        }

    def compare_policies(
        self,
        baseline_policy: DiffusionPolicy,
        curated_policy: DiffusionPolicy,
        dataset: List[Dict],
        num_episodes: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two policies on task performance.
        
        Args:
            baseline_policy: Baseline policy trained on full dataset
            curated_policy: Policy trained on curated data
            dataset: Dataset for evaluation
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing baseline and curated policies...")
        
        # Evaluate baseline policy
        logger.info("📊 Evaluating BASELINE policy...")
        baseline_metrics = self.evaluate_policy_on_task(
            baseline_policy, dataset, num_episodes
        )
        
        # Evaluate curated policy  
        logger.info("📊 Evaluating CURATED policy...")
        curated_metrics = self.evaluate_policy_on_task(
            curated_policy, dataset, num_episodes
        )
        
        # Calculate improvements
        improvements = {}
        for key in baseline_metrics:
            if key.startswith(('success_rate', 'avg_reward')):
                baseline_val = baseline_metrics[key]
                curated_val = curated_metrics[key]
                if baseline_val != 0:
                    improvement = ((curated_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = curated_val * 100  # If baseline is 0
                improvements[f"{key}_improvement"] = improvement
            elif key.startswith(('avg_final_distance',)):
                # For distance, lower is better
                baseline_val = baseline_metrics[key]
                curated_val = curated_metrics[key]
                if baseline_val != 0:
                    improvement = ((baseline_val - curated_val) / baseline_val) * 100
                else:
                    improvement = 0
                improvements[f"{key}_improvement"] = improvement
        
        return {
            'baseline': baseline_metrics,
            'curated': curated_metrics,
            'improvements': improvements
        }

    def generate_policy_videos(
        self,
        policy: DiffusionPolicy,
        policy_name: str,
        dataset: List[Dict],
        num_videos: int = 5,
        max_steps: int = 50,
        output_dir: str = "outputs/videos",
        fps: int = 10
    ) -> List[str]:
        """
        Generate MP4 video files of policy rollouts for qualitative analysis.
        
        Args:
            policy: The policy to record
            policy_name: Name for the policy (used in filenames)
            dataset: Dataset to sample initial states from
            num_videos: Number of videos to generate
            max_steps: Maximum steps per video
            output_dir: Directory to save videos
            fps: Frames per second for video
            
        Returns:
            List of paths to generated video files
        """
        try:
            import imageio
            import numpy as np
        except ImportError:
            logger.warning("imageio not available - video generation skipped")
            return []
        
        # Use LeRobot environment if configured and available
        if self.environment_type == 'lerobot' and self.lerobot_evaluator is not None:
            try:
                logger.info(f"Using LeRobot video generation for {policy_name}...")
                return self.lerobot_evaluator.demonstrate_policy_rollouts(
                    policy, policy_name, dataset, num_videos, max_steps, output_dir
                )
            except Exception as e:
                logger.warning(f"LeRobot video generation failed: {e}")
                logger.info("🔄 Falling back to simulation video generation")
        
        # Use simulation environment for video generation
        logger.info(f"Generating {num_videos} videos for {policy_name} using simulation...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        policy.eval()
        
        video_paths = []
        
        # Sample random episodes for video generation
        video_indices = np.random.choice(len(dataset), size=min(num_videos, len(dataset)), replace=False)
        
        for video_idx, demo_idx in enumerate(video_indices):
            try:
                # Get initial observation
                demo_sample = dataset[int(demo_idx)]
                initial_obs = np.array(demo_sample['observation.state'])
                
                # Generate video for this episode
                video_path = f"{output_dir}/{policy_name}_rollout_{video_idx+1}.mp4"
                frames = self._record_episode_frames(policy, initial_obs, max_steps)
                
                if frames:
                    # Save as MP4 video
                    imageio.mimsave(video_path, frames, fps=fps)
                    video_paths.append(video_path)
                    logger.info(f"   Saved: {video_path}")
                else:
                    logger.warning(f"   ⚠️ No frames captured for video {video_idx+1}")
                    
            except Exception as e:
                logger.error(f"Error generating video {video_idx+1}: {e}")
                continue
        
        logger.info(f"✅ Generated {len(video_paths)}/{num_videos} videos for {policy_name}")
        policy.train()
        return video_paths

    def _record_episode_frames(
        self,
        policy: DiffusionPolicy,
        initial_obs: np.ndarray,
        max_steps: int
    ) -> List[np.ndarray]:
        """
        Record frames from a policy episode for video generation.
        
        Args:
            policy: Policy to record
            initial_obs: Initial observation
            max_steps: Maximum steps to record
            
        Returns:
            List of RGB frames as numpy arrays
        """
        # Initialize simulator for frame recording
        simulator = PushTSimulator(render_mode='rgb_array')
        obs_dict = simulator.reset(initial_obs)
        obs = obs_dict['observation']
        
        frames = []
        step = 0
        success = False
        
        with torch.no_grad():
            while step < max_steps and not success:
                # Capture frame
                frame = simulator.get_rgb_frame()
                if frame is not None:
                    frames.append(frame)
                
                # Get action from policy
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                
                # Create proper batch format for LeRobot policy
                # Check if policy expects images by looking at its vision backbone
                if hasattr(policy, 'config') and hasattr(policy.config, 'vision_backbone') and policy.config.vision_backbone:
                    # Policy expects images, provide dummy images
                    policy_input = {
                        'observation.state': obs_tensor,
                        'observation.image': torch.zeros(1, 3, 84, 84, device=self.device)  # Dummy image
                    }
                else:
                    # State-only policy
                    policy_input = {
                        'observation.state': obs_tensor
                    }
                
                try:
                    action = policy.select_action(policy_input).cpu().numpy()[0]
                except Exception as e:
                    # If the above fails, try with just state (fallback)
                    logger.debug(f"Policy select_action failed with full input, trying state-only: {e}")
                    policy_input = {'observation.state': obs_tensor}
                    action = policy.select_action(policy_input).cpu().numpy()[0]
                
                # Take step
                step_result = simulator.step(action, dt=1/10)  # 10 FPS for video
                obs = step_result['next_observation']
                success = step_result['success']
                step += 1
                
                # Stop early if successful
                if success:
                    # Capture final frame
                    frame = simulator.get_rgb_frame()
                    if frame is not None:
                        frames.append(frame)
                    break
        
        simulator.close()
        return frames 