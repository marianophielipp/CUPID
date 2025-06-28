"""
CUPID: Curating Data your Robot Loves with Influence Functions

Main orchestrator class that coordinates all components for influence-based
demonstration selection in robot imitation learning.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .policy import PolicyManager
from .trainer import PolicyTrainer  
from .influence import InfluenceComputer
from .data import load_trajectories
from .evaluation import TaskEvaluator

logger = logging.getLogger(__name__)


class CUPID:
    """
    CUPID: Curating Data your Robot Loves with Influence Functions
    
    Main class that orchestrates the complete CUPID pipeline for robot
    imitation learning data curation using influence functions.
    
    Based on the paper: "CUPID: Curating Data your Robot Loves with Influence Functions"
    which shows that training with 25-33% of curated data can achieve SOTA performance.
    """
    
    def __init__(self, config: Config, render_mode: Optional[str] = None):
        """
        Initialize CUPID with configuration.
        
        Args:
            config: CUPID configuration object containing all settings
            render_mode: Optional render mode for the TaskEvaluator (e.g., 'human').
        """
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"🚀 Initializing CUPID on device: {self.device}")
        
        # Initialize components
        self.dataset = load_trajectories(
            dataset_name=config.dataset_name,
            max_episodes=config.max_episodes
        )
        self.task_evaluator = TaskEvaluator(config, render_mode=render_mode)
        self.influence_computer = InfluenceComputer(config)
        self.policy_trainer = PolicyTrainer(config)
        self.policy_manager = PolicyManager(config)
        
        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_baseline(self) -> Union[torch.nn.Module, Tuple[torch.nn.Module, List[float]]]:
        """
        Train baseline policy on all demonstrations.
        
        Returns:
            If policy exists: just the trained baseline policy
            If training new: tuple of (trained baseline policy, loss history)
        """
        logger.info("Training baseline policy...")
        
        # Create descriptive checkpoint name with dataset info
        num_trajectories = len(self.dataset)
        flat_dataset = []
        for trajectory in self.dataset:
            flat_dataset.extend(trajectory)
        num_steps = len(flat_dataset)
        
        # Generate checkpoint name with dataset info
        baseline_name = f"baseline_policy_T{num_trajectories}_S{num_steps}.pth"
        baseline_path = Path(self.config.checkpoint_dir) / baseline_name
        
        # Check if exact baseline already exists
        if baseline_path.exists() and not self.config.force_retrain:
            logger.info(f"✅ Loading existing baseline from {baseline_path}")
            logger.info(f"   📊 Dataset: {num_trajectories} trajectories, {num_steps} steps")
            return self.policy_manager.load_policy(baseline_path)
        
        # Check for any baseline with different dataset size
        existing_baselines = list(Path(self.config.checkpoint_dir).glob("baseline_policy_T*_S*.pth"))
        if existing_baselines and not self.config.force_retrain:
            logger.warning(f"⚠️  Found existing baseline(s) with different dataset sizes:")
            for existing in existing_baselines:
                logger.warning(f"   - {existing.name}")
            logger.warning(f"   Current dataset: {num_trajectories} trajectories, {num_steps} steps")
            logger.warning(f"   Use --force-retrain to create new baseline for current dataset")
        
        logger.info(f"🚀 Training new baseline policy with {num_steps} total steps from {num_trajectories} trajectories.")
        
        trained_policy, loss_history = self.policy_trainer.train_policy(
            dataset=flat_dataset, # Pass the flattened dataset to the trainer
            policy_manager=self.policy_manager
        )
        
        # Save baseline with metadata
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_manager.save_policy_with_metadata(
            policy=trained_policy, 
            filepath=baseline_path,
            metadata={
                'policy_type': 'baseline',
                'num_trajectories': num_trajectories,
                'num_steps': num_steps,
                'dataset_name': self.config.dataset_name,
                'training_steps': self.config.training.num_steps,
                'selection_ratio': 1.0,  # Baseline uses all data
                'config_name': getattr(self.config, 'config_name', 'unknown')
            }
        )
        
        logger.info(f"✅ Baseline policy training completed and saved to {baseline_name}")
        return trained_policy, loss_history
    
    def compute_influence_scores(self, policy: torch.nn.Module) -> np.ndarray:
        """
        Compute influence scores for all demonstrations.
        
        Args:
            policy: Trained policy to compute influence against
            
        Returns:
            Array of influence scores
        """
        # Step 1: Collect rollouts from a subset of data
        logger.info(f"Collecting evaluation rollouts...")
        
        # IMPROVED: Use proportional sampling with smart selection
        num_rollouts = self.config.get_eval_sample_count(len(self.dataset))
        
        if num_rollouts >= len(self.dataset):
            # Use all trajectories if we need more than available
            logger.info(f"Using all {len(self.dataset)} trajectories for evaluation rollouts")
            eval_indices = list(range(len(self.dataset)))
        else:
            # Randomly sample without replacement to avoid duplicates
            logger.info(f"Using {num_rollouts}/{len(self.dataset)} trajectories ({num_rollouts/len(self.dataset)*100:.1f}%) for evaluation rollouts")
            eval_indices = np.random.choice(len(self.dataset), num_rollouts, replace=False)
        
        # OPTIMIZED: Pre-allocate rollouts list for better memory performance
        rollouts = [None] * len(eval_indices)
        
        for i, idx in enumerate(tqdm(eval_indices, desc="Collecting rollouts")):
            trajectory = self.dataset[idx]
            
            # Extract initial state from trajectory
            initial_state = np.array(trajectory[0]['observation.state'])
            
            # Run evaluation rollout
            result = self.task_evaluator._run_episode(
                policy, initial_state, max_steps=300
            )
            rollouts[i] = result
        
        logger.info(f"Collected {len(rollouts)} rollouts")
        
        # Step 2: Compute influence scores  
        influence_scores = self.influence_computer.compute_influence_scores(
            policy=policy,
            train_trajectories=self.dataset,
            eval_rollouts=rollouts
        )
        
        return influence_scores
    
    def select_demonstrations(self, influence_scores: np.ndarray) -> List[int]:
        """
        Select trajectory indices based on influence scores.
        
        Args:
            influence_scores: Array of influence scores for each trajectory
            
        Returns:
            List of selected trajectory indices
        """
        selected_indices, selected_scores = self.influence_computer.select_demonstrations(
            influence_scores, len(self.dataset)
        )
        
        return selected_indices.tolist()
    
    def train_curated_policy(self, selected_indices: List[int]) -> Tuple[torch.nn.Module, List[float]]:
        """
        Train policy on a curated subset of demonstrations.
        
        Args:
            selected_indices: Indices of selected trajectories
            
        Returns:
            Tuple of (trained curated policy, loss history)
        """
        # Create a new flat dataset containing only the steps from selected trajectories
        curated_dataset = []
        for idx in selected_indices:
            curated_dataset.extend(self.dataset[idx])
        
        # Calculate dataset statistics
        total_trajectories = len(self.dataset)
        selected_trajectories = len(selected_indices)
        curated_steps = len(curated_dataset)
        selection_ratio = selected_trajectories / total_trajectories
        
        logger.info(f"🚀 Training curated policy with {selected_trajectories} trajectories ({curated_steps} steps)...")
        logger.info(f"   📊 Selection: {selected_trajectories}/{total_trajectories} trajectories ({selection_ratio:.1%})")
        
        # Train policy with the curated subset of steps
        trained_policy, loss_history = self.policy_trainer.train_policy(
            dataset=curated_dataset,
            policy_manager=self.policy_manager
        )
        
        # Generate descriptive checkpoint name
        curated_name = f"curated_policy_T{selected_trajectories}of{total_trajectories}_S{curated_steps}_{selection_ratio:.0%}.pth"
        checkpoint_path = Path(self.config.checkpoint_dir) / curated_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save curated policy with metadata
        self.policy_manager.save_policy_with_metadata(
            policy=trained_policy,
            filepath=checkpoint_path,
            metadata={
                'policy_type': 'curated',
                'num_trajectories': selected_trajectories,
                'num_steps': curated_steps,
                'total_available_trajectories': total_trajectories,
                'selection_ratio': selection_ratio,
                'selected_indices': selected_indices,
                'dataset_name': self.config.dataset_name,
                'training_steps': self.config.training.num_steps,
                'config_name': getattr(self.config, 'config_name', 'unknown')
            }
        )
        
        logger.info(f"✅ Curated policy training completed and saved to {curated_name}")
        return trained_policy, loss_history
    
    def run_full_pipeline(self) -> torch.nn.Module:
        """
        Run complete CUPID pipeline: train baseline, compute influences, select demos, train curated.
        
        Returns:
            Final trained curated policy
        """
        logger.info("Running complete CUPID pipeline...")
        
        # Step 1: Train baseline
        baseline_policy = self.train_baseline()
        
        # Step 2: Compute influence scores  
        influence_scores = self.compute_influence_scores(baseline_policy)
        
        # Step 3: Select demonstrations
        selected_indices = self.select_demonstrations(influence_scores)
        
        # Step 4: Train curated policy
        curated_policy, _ = self.train_curated_policy(selected_indices)
        
        logger.info("Complete CUPID pipeline finished")
        return curated_policy
    
    def evaluate_policy_on_task(self, policy: torch.nn.Module, 
                               num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate policy on actual task performance (success rate, rewards, etc.).
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to evaluate. If None, uses config value.
            
        Returns:
            Dictionary of task performance metrics
        """
        if num_episodes is None:
            num_episodes = self.config.evaluation.num_episodes
            
        return self.task_evaluator.evaluate_policy_on_task(
            policy, self.dataset, num_episodes
        )
    
    def compare_policies(self, baseline_policy: torch.nn.Module,
                        curated_policy: torch.nn.Module,
                        num_episodes: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare baseline and curated policies on task performance.
        
        Args:
            baseline_policy: Baseline policy trained on all data
            curated_policy: Policy trained on curated data
            num_episodes: Number of episodes for evaluation. If None, uses config value.
            
        Returns:
            Dictionary with comparison metrics
        """
        if num_episodes is None:
            num_episodes = self.config.evaluation.num_episodes
        
        # Flatten trajectories for evaluation (evaluator expects individual steps)
        flat_dataset = []
        for trajectory in self.dataset:
            flat_dataset.extend(trajectory)
            
        return self.task_evaluator.compare_policies(
            baseline_policy, curated_policy, flat_dataset, num_episodes
        )
    
    def evaluate_policy(self, policy: torch.nn.Module, 
                       num_samples: int = 200) -> Dict[str, float]:
        """
        Evaluate policy performance on dataset (legacy method for backward compatibility).
        
        Args:
            policy: Policy to evaluate
            num_samples: Number of samples to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating policy on {num_samples} samples...")
        
        metrics = self.policy_trainer.evaluate_policy(policy, self.dataset, num_samples)
        
        logger.info(f"Evaluation completed: {metrics}")
        return metrics
    
    def get_influence_statistics(self, influence_scores: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about influence scores.
        
        Args:
            influence_scores: Array of influence scores
            
        Returns:
            Dictionary of statistics
        """
        return {
            "mean": float(np.mean(influence_scores)),
            "std": float(np.std(influence_scores)),
            "min": float(np.min(influence_scores)),
            "max": float(np.max(influence_scores)),
            "median": float(np.median(influence_scores)),
            "q25": float(np.percentile(influence_scores, 25)),
            "q75": float(np.percentile(influence_scores, 75))
        }
    
    def save_influence_scores(self, influence_scores: np.ndarray, 
                            filepath: Optional[str] = None) -> None:
        """
        Save influence scores to file.
        
        Args:
            influence_scores: Array of influence scores
            filepath: Path to save file. If None, uses default path.
        """
        if filepath is None:
            filepath = Path(self.config.checkpoint_dir) / "influence_scores.npy"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(filepath, influence_scores)
        logger.info(f"Saved influence scores to {filepath}")
    
    def load_influence_scores(self, filepath: Optional[str] = None) -> np.ndarray:
        """
        Load influence scores from file.
        
        Args:
            filepath: Path to load file. If None, uses default path.
            
        Returns:
            Array of influence scores
        """
        if filepath is None:
            filepath = Path(self.config.checkpoint_dir) / "influence_scores.npy"
            
        influence_scores = np.load(filepath)
        logger.info(f"Loaded influence scores from {filepath}")
        return influence_scores 