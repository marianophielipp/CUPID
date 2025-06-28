"""
Dataset management for CUPID.

This module provides dataset loading and management functionality for robot imitation learning.
"""

import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
import torch
import numpy as np
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)


def load_trajectories(dataset_name: str, max_episodes: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """
    Load a dataset from HuggingFace and group it into trajectories.

    Args:
        dataset_name: The name of the dataset on HuggingFace Hub.
        max_episodes: The maximum number of episodes to load. If None, all are loaded.

    Returns:
        A list of trajectories, where each trajectory is a list of steps.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    if max_episodes is not None and max_episodes > 0:
        # This is an approximation. We find the step index that corresponds to the
        # end of the `max_episodes-1`-th episode.
        if 'episode_index' in dataset.features:
            episode_indices = np.array(dataset['episode_index'])
            if max_episodes < np.max(episode_indices):
                cutoff_index = np.where(episode_indices == max_episodes)[0][0]
                dataset = dataset.select(range(cutoff_index))

    logger.info(f"Loaded {len(dataset)} total steps.")

    if 'episode_index' not in dataset.features:
        logger.warning("'episode_index' not found. Treating each step as a separate trajectory.")
        return [[sample] for sample in dataset]

    logger.info("Grouping dataset into trajectories...")
    
    trajectories = []
    current_trajectory = []
    
    # Use numpy for efficient grouping
    episode_indices = np.array(dataset['episode_index'])
    # Find the indices where the episode changes
    split_indices = np.where(np.diff(episode_indices) != 0)[0] + 1
    
    # Split the dataset into trajectories based on the indices
    episode_datasets = np.split(dataset, split_indices)
    
    for episode_data in tqdm(episode_datasets, desc="Grouping trajectories"):
        trajectories.append(list(episode_data))

    logger.info(f"Grouped data into {len(trajectories)} trajectories.")
    return trajectories


class DatasetManager:
    """
    Manages dataset loading and manipulation for CUPID.
    
    Handles loading datasets from HuggingFace, creating subsets,
    and preparing data for training and influence computation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize DatasetManager.
        
        Args:
            config: CUPID configuration object
        """
        self.config = config
        self.device = config.device
        
    def load_dataset(self) -> Dataset:
        """
        Load dataset from HuggingFace.
        
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load from HuggingFace
        dataset = load_dataset(self.config.dataset_name, split="train")
        
        # Limit episodes if specified
        if self.config.max_episodes:
            max_samples = min(len(dataset), self.config.max_episodes)
            dataset = dataset.select(range(max_samples))
            
        logger.info(f"Loaded {len(dataset)} demonstrations")
        
        # Log dataset statistics
        if len(dataset) > 0:
            sample = dataset[0]
            if 'observation.state' in sample:
                obs_dim = len(sample['observation.state'])
                logger.info(f"Observation dimension: {obs_dim}")
            if 'action' in sample:
                action_dim = len(sample['action'])
                logger.info(f"Action dimension: {action_dim}")
        
        return dataset
    
    def create_subset(self, dataset: Dataset, indices: List[int]) -> Dataset:
        """
        Create a subset of the dataset using specified indices.
        
        Args:
            dataset: Original dataset
            indices: List of indices to include in subset
            
        Returns:
            Subset dataset
        """
        if not indices:
            raise ValueError("Cannot create subset with empty indices")
            
        # Ensure indices are valid
        max_idx = len(dataset) - 1
        valid_indices = [idx for idx in indices if 0 <= idx <= max_idx]
        
        if len(valid_indices) != len(indices):
            logger.warning(f"Filtered {len(indices) - len(valid_indices)} invalid indices")
            
        subset = dataset.select(valid_indices)
        logger.info(f"Created subset with {len(subset)} demonstrations")
        
        return subset
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        if len(dataset) == 0:
            return {"num_episodes": 0}
            
        stats = {
            "num_episodes": len(dataset),
            "features": list(dataset.features.keys())
        }
        
        # Analyze first sample for dimensions
        sample = dataset[0]
        
        if 'observation.state' in sample:
            obs = sample['observation.state']
            stats["observation_dim"] = len(obs) if hasattr(obs, '__len__') else 1
            
        if 'action' in sample:
            action = sample['action']
            stats["action_dim"] = len(action) if hasattr(action, '__len__') else 1
            
        # Analyze episode lengths efficiently using numpy
        if 'episode_index' in dataset.features:
            episode_indices = np.array(dataset['episode_index'])
            if len(episode_indices) > 0:
                _, counts = np.unique(episode_indices, return_counts=True)
                stats["episode_lengths"] = {
                    "mean": float(np.mean(counts)),
                    "std": float(np.std(counts)),
                    "min": int(np.min(counts)),
                    "max": int(np.max(counts)),
                }
        
        return stats
    
    def convert_to_tensors(self, dataset: Dataset) -> List[Dict[str, torch.Tensor]]:
        """
        Convert dataset to list of tensor dictionaries for training.
        
        OPTIMIZED: Using vectorized operations where possible.
        
        Args:
            dataset: Dataset to convert
            
        Returns:
            List of dictionaries with tensor data
        """
        if len(dataset) == 0:
            return []
        
        # Pre-allocate list for better memory performance
        tensor_data = []
        tensor_data_reserve = len(dataset)
        
        # OPTIMIZED: Process items more efficiently
        for item in dataset:
            tensor_item = {}
            
            # OPTIMIZED: Handle common tensor conversions more efficiently
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    # Direct numpy array conversion
                    tensor_item[key] = torch.from_numpy(value).float().to(self.device)
                elif isinstance(value, list):
                    # List to tensor conversion
                    tensor_item[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
                else:
                    # Single value conversion
                    tensor_item[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
                    
            tensor_data.append(tensor_item)
            
        logger.info(f"Converted {len(tensor_data)} items to tensors")
        return tensor_data
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate that dataset has required fields for CUPID.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if valid, False otherwise
        """
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
            
        required_fields = ['observation.state', 'action']
        sample = dataset[0]
        
        for field in required_fields:
            if field not in sample:
                logger.error(f"Missing required field: {field}")
                return False
                
        logger.info("Dataset validation passed")
        return True
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
        """
        Split dataset into training and validation sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Fraction of data for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset 