"""
CUPID: Curating Data your Robot Loves with Influence Functions

A modular implementation for robot imitation learning data curation using influence functions.
Based on the paper by Agia et al. that demonstrates how to identify which demonstrations
truly matter for policy performance.

Key components:
- Config: Configuration management
- CUPID: Main orchestrator class  
- PolicyManager: Policy creation and management
- Trainer: Training with LeRobot alignment
- InfluenceComputer: Influence function computation
- DatasetManager: Dataset loading and management
- TaskEvaluator: Task-based performance evaluation
- Visualization: Clean visualizations for results analysis
"""

from .config import Config, TrainingConfig, PolicyConfig, InfluenceConfig
from .cupid import CUPID
from .policy import PolicyManager, DiffusionPolicy
from .trainer import Trainer
from .influence import InfluenceComputer
from .data import DatasetManager
from .evaluation import TaskEvaluator
from .visualization import CUPIDVisualizer, create_cupid_visualization

__all__ = [
    # Core components
    'Config', 'TrainingConfig', 'PolicyConfig', 'InfluenceConfig',
    'CUPID',
    'PolicyManager', 'DiffusionPolicy',
    'Trainer',
    'InfluenceComputer', 
    'DatasetManager',
    'TaskEvaluator',
    
    # Visualization
    'CUPIDVisualizer', 'create_cupid_visualization'
]

__version__ = "2.0.0" 