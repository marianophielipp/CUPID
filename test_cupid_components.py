#!/usr/bin/env python3
"""
Test script to verify CUPID components work correctly.
This provides a minimal test of the core functionality.
"""

import torch
import numpy as np
from src.cupid import CUPID, Config
from src.cupid.policy import DiffusionPolicy, PolicyManager
from src.cupid.influence import InfluenceComputer
from src.cupid.data import load_trajectories
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_policy_forward_pass():
    """Test that the diffusion policy can do a forward pass."""
    logger.info("Testing Diffusion Policy forward pass...")
    
    config = Config.smoke_test()
    policy_manager = PolicyManager(config)
    policy = policy_manager.create_policy()
    
    # Create dummy inputs
    batch_size = 4
    obs = torch.randn(batch_size, 2)  # 2D state observation
    noisy_actions = torch.randn(batch_size, 2)  # 2D actions
    timestep = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    noise_pred = policy(obs, noisy_actions, timestep)
    
    assert noise_pred.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {noise_pred.shape}"
    logger.info("✅ Policy forward pass successful")
    

def test_policy_sampling():
    """Test that the policy can sample actions."""
    logger.info("Testing Diffusion Policy action sampling...")
    
    config = Config.smoke_test()
    policy_manager = PolicyManager(config)
    policy = policy_manager.create_policy()
    
    # Create dummy observation
    batch_size = 2
    obs = torch.randn(batch_size, 2)
    
    # Sample actions
    actions = policy.sample_action(obs)
    
    assert actions.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {actions.shape}"
    logger.info("✅ Policy action sampling successful")


def test_influence_computation():
    """Test influence score computation with dummy data."""
    logger.info("Testing Influence Score computation...")
    
    config = Config.smoke_test()
    influence_computer = InfluenceComputer(config)
    
    # Create dummy trajectories
    num_trajectories = 5
    trajectory_length = 10
    
    trajectories = []
    for i in range(num_trajectories):
        trajectory = []
        for t in range(trajectory_length):
            step = {
                'observation.state': np.random.randn(2).astype(np.float32),
                'action': np.random.randn(2).astype(np.float32)
            }
            trajectory.append(step)
        trajectories.append(trajectory)
    
    # Create dummy policy
    policy_manager = PolicyManager(config)
    policy = policy_manager.create_policy()
    
    # Create dummy rollouts
    rollouts = []
    for i in range(3):
        rollout = {
            'trajectory': [
                {
                    'observation': np.random.randn(2).astype(np.float32),
                    'action': np.random.randn(2).astype(np.float32)
                }
                for _ in range(10)
            ],
            'reward': np.random.rand()
        }
        rollouts.append(rollout)
    
    # Compute influence scores
    influence_scores = influence_computer.compute_influence_scores(
        policy=policy,
        train_trajectories=trajectories,
        eval_rollouts=rollouts
    )
    
    assert len(influence_scores) == num_trajectories, f"Expected {num_trajectories} scores, got {len(influence_scores)}"
    assert np.all(np.isfinite(influence_scores)), "Influence scores contain NaN or Inf"
    logger.info(f"✅ Influence computation successful. Scores: {influence_scores}")


def test_demonstration_selection():
    """Test demonstration selection based on influence scores."""
    logger.info("Testing Demonstration Selection...")
    
    config = Config.smoke_test()
    influence_computer = InfluenceComputer(config)
    
    # Create dummy influence scores
    influence_scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8, 0.6, 0.15])
    dataset_size = len(influence_scores)
    
    # Select demonstrations
    selected_indices, selected_scores = influence_computer.select_demonstrations(
        influence_scores, dataset_size
    )
    
    # Check selection
    expected_count = config.get_selection_count(dataset_size)
    assert len(selected_indices) == expected_count, f"Expected {expected_count} selections, got {len(selected_indices)}"
    
    # Check that selected scores are the highest
    assert np.all(selected_scores == np.sort(influence_scores)[::-1][:expected_count])
    logger.info(f"✅ Demonstration selection successful. Selected indices: {selected_indices}")


def test_cupid_integration():
    """Test basic CUPID integration."""
    logger.info("Testing CUPID integration...")
    
    config = Config.smoke_test(max_episodes=10)
    
    # Create minimal dataset
    trajectories = []
    for i in range(10):
        trajectory = []
        for t in range(5):
            step = {
                'observation.state': np.random.randn(2).astype(np.float32),
                'action': np.random.randn(2).astype(np.float32),
                'episode_index': i,
                'frame_index': t
            }
            trajectory.append(step)
        trajectories.append(trajectory)
    
    # Mock the load_trajectories function
    import src.cupid.data as data_module
    original_load = data_module.load_trajectories
    data_module.load_trajectories = lambda dataset_name, max_episodes: trajectories
    
    try:
        # Initialize CUPID
        cupid = CUPID(config)
        
        # Check dataset loaded
        assert len(cupid.dataset) == 10, f"Expected 10 trajectories, got {len(cupid.dataset)}"
        logger.info("✅ CUPID initialization successful")
        
    finally:
        # Restore original function
        data_module.load_trajectories = original_load


def main():
    """Run all tests."""
    logger.info("Starting CUPID component tests...")
    
    tests = [
        test_policy_forward_pass,
        test_policy_sampling,
        test_influence_computation,
        test_demonstration_selection,
        test_cupid_integration
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {str(e)}")
            failed += 1
    
    if failed == 0:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.error(f"\n❌ {failed}/{len(tests)} tests failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())