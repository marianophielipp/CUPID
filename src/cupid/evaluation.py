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
        
        # Reset trajectory
        self.trajectory = []
        
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
        """Initialize pygame rendering for Mac."""
        try:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("PushT Simulator")
            self.clock = pygame.time.Clock()
            logger.info("✅ Rendering enabled for cross-platform compatibility")
        except ImportError:
            logger.warning("⚠️ pygame not available, rendering disabled")
            self.render_mode = None
    
    def render(self):
        """Render the environment state (matching LeRobot PushT style)."""
        if self.screen is None:
            return
            
        try:
            import pygame
            
            # Clear screen with workspace color
            self.screen.fill((50, 50, 50))  # Dark workspace like real PushT
            
            # Scale positions to window size - now using [0,512] workspace
            scale_x = self.window_size / (self.workspace_bounds[1] - self.workspace_bounds[0])
            scale_y = self.window_size / (self.workspace_bounds[3] - self.workspace_bounds[2])
            
            # Draw workspace boundary
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (20, 20, self.window_size - 40, self.window_size - 40), 2)
            
            # Draw goal zone (large green circle)
            goal_x = int((self.goal_zone_center[0] - self.workspace_bounds[0]) * scale_x)
            goal_y = int((self.goal_zone_center[1] - self.workspace_bounds[2]) * scale_y)
            goal_radius = int(self.goal_zone_size / 2 * scale_x)
            
            # Goal zone background (semi-transparent green)
            goal_surface = pygame.Surface((goal_radius * 2, goal_radius * 2))
            goal_surface.set_alpha(64)  # Semi-transparent
            goal_surface.fill((0, 255, 0))
            pygame.draw.circle(goal_surface, (0, 255, 0), (goal_radius, goal_radius), goal_radius)
            self.screen.blit(goal_surface, (goal_x - goal_radius, goal_y - goal_radius))
            
            # Goal zone outline
            pygame.draw.circle(self.screen, (0, 200, 0), (goal_x, goal_y), goal_radius, 3)
            
            # Draw target T-shape in goal zone (green, showing desired final position)
            target_t_size = int(self.object_size / 2 * scale_x)
            target_horizontal_width = target_t_size * 2
            target_horizontal_height = 12
            target_vertical_width = 16
            target_vertical_height = int(target_t_size * 1.2)
            
            # Target T-shape (bright green, semi-transparent)
            target_surface = pygame.Surface((target_horizontal_width, target_horizontal_height))
            target_surface.set_alpha(180)  # Semi-transparent
            target_surface.fill((0, 255, 0))
            self.screen.blit(target_surface, (goal_x - target_horizontal_width//2, goal_y - target_horizontal_height//2))
            
            target_surface_v = pygame.Surface((target_vertical_width, target_vertical_height))
            target_surface_v.set_alpha(180)  # Semi-transparent
            target_surface_v.fill((0, 255, 0))
            self.screen.blit(target_surface_v, (goal_x - target_vertical_width//2, goal_y - target_horizontal_height//2))
            
            # Target T-shape outline (darker green)
            pygame.draw.rect(self.screen, (0, 150, 0), 
                           (goal_x - target_horizontal_width//2, goal_y - target_horizontal_height//2, 
                            target_horizontal_width, target_horizontal_height), 2)
            pygame.draw.rect(self.screen, (0, 150, 0), 
                           (goal_x - target_vertical_width//2, goal_y - target_horizontal_height//2, 
                            target_vertical_width, target_vertical_height), 2)
            
            # Draw T-shaped object (bright cyan/blue filled) - more realistic proportions
            obj_x = int((self.object_position[0] - self.workspace_bounds[0]) * scale_x)
            obj_y = int((self.object_position[1] - self.workspace_bounds[2]) * scale_y)
            
            # Better T-shape proportions (like real PushT)
            t_size = int(self.object_size / 2 * scale_x)
            horizontal_width = t_size * 2  # Full width of horizontal bar
            horizontal_height = 12  # Height of horizontal bar
            vertical_width = 16  # Width of vertical bar  
            vertical_height = int(t_size * 1.2)  # Height of vertical bar (extends down)
            
            # Current T-shape (filled bright cyan) - horizontal bar (top)
            pygame.draw.rect(self.screen, (0, 200, 255), 
                           (obj_x - horizontal_width//2, obj_y - horizontal_height//2, 
                            horizontal_width, horizontal_height))
            
            # Vertical bar of T (extends downward from center of horizontal bar)
            pygame.draw.rect(self.screen, (0, 200, 255), 
                           (obj_x - vertical_width//2, obj_y - horizontal_height//2, 
                            vertical_width, vertical_height))
            
            # T-shape outline (darker blue) - horizontal bar
            pygame.draw.rect(self.screen, (0, 100, 200), 
                           (obj_x - horizontal_width//2, obj_y - horizontal_height//2, 
                            horizontal_width, horizontal_height), 2)
            
            # T-shape outline - vertical bar
            pygame.draw.rect(self.screen, (0, 100, 200), 
                           (obj_x - vertical_width//2, obj_y - horizontal_height//2, 
                            vertical_width, vertical_height), 2)
            
            # Draw circular pusher (bright yellow) - the robot end-effector
            pusher_x = int((self.pusher_position[0] - self.workspace_bounds[0]) * scale_x)
            pusher_y = int((self.pusher_position[1] - self.workspace_bounds[2]) * scale_y)
            pusher_radius_px = int(self.pusher_radius * scale_x)
            
            # Check if pusher is making contact
            contact_distance = np.linalg.norm(self.pusher_position - self.object_position)
            makes_contact = contact_distance < (self.pusher_radius + self.object_size / 2)
            
            # Color pusher based on contact
            pusher_color = (255, 100, 100) if makes_contact else (255, 255, 0)  # Red if contact, yellow otherwise
            
            pygame.draw.circle(self.screen, pusher_color, (pusher_x, pusher_y), pusher_radius_px)
            pygame.draw.circle(self.screen, (200, 200, 0), (pusher_x, pusher_y), pusher_radius_px, 2)
            
            # Draw pusher target position (where action wants pusher to go)
            if hasattr(self, 'last_action') and self.last_action is not None:
                action_x = int((self.last_action[0] - self.workspace_bounds[0]) * scale_x)
                action_y = int((self.last_action[1] - self.workspace_bounds[2]) * scale_y)
                # Draw target as small cross
                pygame.draw.line(self.screen, (255, 150, 0), 
                               (action_x - 5, action_y), (action_x + 5, action_y), 2)
                pygame.draw.line(self.screen, (255, 150, 0), 
                               (action_x, action_y - 5), (action_x, action_y + 5), 2)
                # Draw line from pusher to target
                pygame.draw.line(self.screen, (255, 150, 0), 
                               (pusher_x, pusher_y), (action_x, action_y), 1)
            
            # Draw contact visualization
            if makes_contact:
                # Draw contact region
                contact_radius = int((self.pusher_radius + self.object_size / 2) * scale_x)
                pygame.draw.circle(self.screen, (255, 100, 100), (pusher_x, pusher_y), contact_radius, 1)
                # Draw velocity vector if object is moving
                if np.linalg.norm(self.object_velocity) > 0.5:
                    velocity_end_x = obj_x + int(self.object_velocity[0] * scale_x * 2)
                    velocity_end_y = obj_y + int(self.object_velocity[1] * scale_y * 2)
                    pygame.draw.line(self.screen, (255, 0, 0), 
                                   (obj_x, obj_y), (velocity_end_x, velocity_end_y), 3)
            
            # Draw distance line to goal zone center
            pygame.draw.line(self.screen, (128, 128, 128), (obj_x, obj_y), (goal_x, goal_y), 1)
            
            # Add comprehensive text overlay
            font = pygame.font.Font(None, 24)
            distance = np.linalg.norm(self.object_position - self.goal_zone_center)
            text = font.render(f"Distance: {distance:.1f}px", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            
            # Show contact status
            contact_text = "CONTACT!" if makes_contact else "No Contact"
            contact_color = (255, 100, 100) if makes_contact else (255, 255, 255)
            contact_status = font.render(contact_text, True, contact_color)
            self.screen.blit(contact_status, (10, 35))
            
            # Show coverage (most important metric)
            coverage = self._calculate_goal_coverage()
            coverage_text = f"Coverage: {coverage:.1%}"
            coverage_color = (255, 255, 0) if coverage >= self.success_threshold else (255, 255, 255)
            coverage_status = font.render(coverage_text, True, coverage_color)
            self.screen.blit(coverage_status, (10, 60))
            
            # Show success status
            success_text = "SUCCESS!" if coverage >= self.success_threshold else "Pushing..."
            color = (0, 255, 0) if coverage >= self.success_threshold else (255, 255, 255)
            status = font.render(success_text, True, color)
            self.screen.blit(status, (10, 85))
            
            # Show action info
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
            
            pygame.display.flip()
            
            # Control frame rate
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self.clock.tick(20)  # 20 FPS for smooth rendering
            
        except ImportError:
            logger.warning("pygame not available for rendering")
        except Exception as e:
            logger.error(f"Rendering error: {e}")
    
    def close(self):
        """Close the rendering window."""
        if self.screen is not None:
            import pygame
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
        self.simulator = PushTSimulator(render_mode=render_mode)
        
        # Try to initialize LeRobot evaluator (quick check)
        self.lerobot_evaluator = None
        try:
            # Quick check if LeRobot is available
            import lerobot
            from .lerobot_env import LeRobotEnvironmentEvaluator
            self.lerobot_evaluator = LeRobotEnvironmentEvaluator(config)
            logger.info("✅ LeRobot available for evaluation (will verify during actual use)")
        except Exception as e:
            logger.info(f"ℹ️ LeRobot not available, will use simulation ({str(e)[:50]}...)")
            self.lerobot_evaluator = None
    
    def evaluate_policy_on_task(
        self,
        policy: DiffusionPolicy,
        dataset: List[Dict],
        num_episodes: int = 100,
        max_steps_per_episode: int = 50,
        use_lerobot: bool = False  # Disabled by default, enable only if LeRobot is properly set up
    ) -> Dict[str, float]:
        """
        Evaluate policy on actual task performance using LeRobot environment or simulation.
        
        Args:
            policy: Policy to evaluate
            dataset: Dataset containing demonstrations
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode (for simulation)
            use_lerobot: Whether to try using real LeRobot environment first
            
        Returns:
            Dictionary with task performance metrics
        """
        logger.info(f"Evaluating policy on {num_episodes} task episodes...")
        
        # Try LeRobot evaluation first if available and requested
        if use_lerobot and self.lerobot_evaluator is not None:
            try:
                logger.info("🤖 Attempting LeRobot PushT environment (will fallback to simulation if issues)...")
                
                # Quick check for LeRobot availability
                try:
                    import lerobot.common.envs.env_utils
                    logger.info("✅ LeRobot components available, proceeding...")
                except ImportError:
                    logger.info("❌ LeRobot components missing, using simulation fallback")
                    raise ImportError("LeRobot components not available")
                
                metrics = self.lerobot_evaluator.evaluate_policy_with_lerobot(policy, num_episodes)
                
                # Add some derived metrics for compatibility
                if 'action_consistency' not in metrics:
                    metrics['action_consistency'] = max(0.0, metrics.get('avg_reward', 0.0))
                
                logger.info(f"✅ LeRobot evaluation complete:")
                logger.info(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
                logger.info(f"  Average reward: {metrics.get('avg_reward', 0):.3f}")
                
                return metrics
                
            except Exception as e:
                logger.info(f"🔄 LeRobot not available ({str(e)[:100]}...), using simulation fallback")
        
        # Fallback to simulation evaluation
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
                episode_result = self._run_episode(
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
        Run a single episode with the policy.
        
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
            current_obs = torch.FloatTensor(state['observation']).unsqueeze(0).to(self.device)
            
            # Generate action with policy using proper diffusion sampling
            predicted_action = policy.sample_action(current_obs)
            
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
            'action_consistency': 1.0 - (np.mean(final_distances) / 400.0),  # Normalized consistency
            'task_efficiency': np.mean(successes) / max(np.mean(episode_lengths), 1)
        }
    
    def demonstrate_policy_rollouts(
        self,
        policy: DiffusionPolicy,
        policy_name: str,
        dataset: List[Dict],
        num_rollouts: int = 10,
        max_steps: int = 50
    ) -> None:
        """
        Demonstrate policy rollouts with interactive controls.
        
        Args:
            policy: The policy to demonstrate
            policy_name: Name for display purposes
            dataset: Dataset to sample initial states from
            num_rollouts: Number of rollouts to show
            max_steps: Maximum steps per rollout
        """
        if self.render_mode != 'human':
            logger.info(f"Rendering disabled, skipping {policy_name} demonstrations")
            return
            
        logger.info(f"🎥 Demonstrating {policy_name} rollouts...")
        
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
        
        logger.info(f"      🎬 Interactive animated rollout...")
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
                    action = policy.sample_action(obs_tensor).cpu().numpy()[0]
                
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
        
        # Evaluate both policies
        baseline_metrics = self.evaluate_policy_on_task(
            baseline_policy, dataset, num_episodes
        )
        curated_metrics = self.evaluate_policy_on_task(
            curated_policy, dataset, num_episodes
        )
        
        # Calculate improvements
        improvements = {}
        for key in baseline_metrics:
            if key.startswith(('success_rate', 'avg_reward', 'action_consistency')):
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