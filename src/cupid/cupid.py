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
from .data import DatasetManager
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
    
    def __init__(self, config: Config):
        """
        Initialize CUPID with configuration.
        
        Args:
            config: CUPID configuration object containing all settings
        """
        self.config = config
        
        # Initialize components
        self.dataset_manager = DatasetManager(config)
        self.policy_manager = PolicyManager(config)
        self.trainer = PolicyTrainer(config)
        self.influence_computer = InfluenceComputer(config)
        self.task_evaluator = TaskEvaluator(config)
        
        # Load dataset and group into trajectories
        logger.info(f"Loading and processing dataset: {config.dataset_name}")
        self.dataset = self.dataset_manager.load_dataset()
        self.trajectories = self.dataset_manager.load_trajectories(self.dataset)
        logger.info(f"Dataset loaded with {len(self.dataset)} steps across {len(self.trajectories)} trajectories.")
    
    def train_baseline(self) -> Union[torch.nn.Module, Tuple[torch.nn.Module, List[float]]]:
        """
        Train baseline policy on all demonstrations.
        
        Returns:
            If policy exists: just the trained baseline policy
            If training new: tuple of (trained baseline policy, loss history)
        """
        logger.info("Training baseline policy...")
        
        # Check if baseline already exists
        baseline_path = Path(self.config.checkpoint_dir) / "baseline_policy.pth"
        
        if baseline_path.exists() and not self.config.force_retrain:
            logger.info(f"Loading existing baseline from {baseline_path}")
            return self.policy_manager.load_policy(baseline_path)
        
        # Train new baseline. The trainer still expects a flat list of steps.
        logger.info(f"Training new baseline policy with {len(self.dataset)} total steps.")
        
        trained_policy, loss_history = self.trainer.train_policy(
            dataset=self.dataset, # Pass the flat dataset to the trainer
            policy_manager=self.policy_manager
        )
        
        # Save baseline
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_manager.save_policy(trained_policy, baseline_path)
        
        logger.info("Baseline policy training completed")
        return trained_policy, loss_history
    
    def compute_influence_scores(self, policy: torch.nn.Module) -> np.ndarray:
        """
        Compute trajectory-based influence scores using the proper CUPID algorithm.
        
        Args:
            policy: Trained policy to use for influence computation
            
        Returns:
            Array of influence scores for each trajectory
        """
        logger.info("Computing trajectory-based influence scores...")
        
        # Collect policy rollouts for performance influence computation
        logger.info("Collecting policy rollouts for performance gradient...")
        rollouts = []
        
        # Use configured number of samples for rollouts
        num_rollouts = self.config.influence.num_samples
        max_steps = 50  # Reasonable default for PushT rollouts
        
        if len(self.dataset) > num_rollouts:
            eval_indices = np.random.choice(len(self.dataset), num_rollouts, replace=False)
        else:
            eval_indices = np.arange(len(self.dataset))

        for idx in tqdm(eval_indices, desc="Collecting rollouts"):
            demo = self.dataset[int(idx)]
            initial_state = np.array(demo['observation.state'])
            
            # Run policy rollout using task evaluator
            result = self.task_evaluator._run_episode(
                policy, 
                initial_state,
                max_steps=max_steps
            )
            rollouts.append(result)
        
        logger.info(f"Collected {len(rollouts)} rollouts.")
        
        # Compute trajectory-based influence scores
        influence_scores = self.influence_computer.compute_influence_scores(
            policy=policy,
            train_trajectories=self.trajectories,
            eval_rollouts=rollouts
        )
        
        logger.info(f"Computed influence scores for {len(influence_scores)} trajectories.")
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
            influence_scores, len(self.trajectories)
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
            curated_dataset.extend(self.trajectories[idx])
            
        logger.info(f"Training curated policy with {len(selected_indices)} trajectories ({len(curated_dataset)} steps)...")
        
        # Train policy with the curated subset of steps
        trained_policy, loss_history = self.trainer.train_policy(
            dataset=curated_dataset,
            policy_manager=self.policy_manager
        )
        
        # Save curated policy
        selection_ratio = len(selected_indices) / len(self.trajectories)
        checkpoint_path = Path(self.config.checkpoint_dir) / f"curated_policy_{selection_ratio:.0%}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_manager.save_policy(trained_policy, checkpoint_path)
        
        logger.info("Curated policy training completed")
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
            
        return self.task_evaluator.compare_policies(
            baseline_policy, curated_policy, self.dataset, num_episodes
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
        
        metrics = self.trainer.evaluate_policy(policy, self.dataset, num_samples)
        
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