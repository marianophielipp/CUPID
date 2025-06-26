#!/usr/bin/env python3
"""
CUPID Example: Complete workflow demonstration

This example shows how to use CUPID for robot imitation learning data curation.
Based on the paper "CUPID: Curating Data your Robot Loves with Influence Functions"
which demonstrates that training with ~25-33% of curated data can achieve state-of-the-art performance.
"""

import torch
from pathlib import Path
import numpy as np
import logging
import sys
import argparse
from typing import Optional
from dataclasses import asdict

# Configure logging for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cupid.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from cupid import CUPID, Config
    from cupid.visualization import create_cupid_visualization
except ImportError as e:
    logger.error(f"Failed to import CUPID modules: {e}")
    logger.error("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)


def setup_error_handling():
    """Setup global error handling for production use."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("🚪 CUPID workflow interrupted by user")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error(
            "💥 Uncaught exception occurred",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception


def validate_environment():
    """Validate that the environment is properly set up."""
    logger.info("🔍 Validating environment...")
    
    # Check PyTorch installation
    try:
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            logger.info(f"✅ MPS available: {torch.backends.mps.is_available()}")
        else:
            logger.info("ℹ️ MPS not available (requires PyTorch 1.12+)")
    except Exception as e:
        logger.error(f"❌ PyTorch validation failed: {e}")
        return False
    
    # Check optional dependencies
    try:
        import pygame
        logger.info(f"✅ pygame version: {pygame.version.ver}")
    except ImportError:
        logger.warning("⚠️ pygame not available - rendering will be disabled")
    
    try:
        import datasets
        logger.info(f"✅ datasets version: {datasets.__version__}")
    except ImportError:
        logger.error("❌ HuggingFace datasets not available - required for data loading")
        return False
    
    logger.info("✅ Environment validation complete")
    return True


def main(render=False, max_episodes=None, config_name="quick_demo"):
    """
    Complete CUPID workflow example with comprehensive error handling.
    
    Args:
        render: Enable visual rendering
        max_episodes: Maximum number of episodes to use
        config_name: Configuration preset name
    """
    
    # Setup error handling
    setup_error_handling()
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed - exiting")
        return 1
    
    logger.info("🤖 CUPID: Curating Data your Robot Loves with Influence Functions")
    logger.info("=" * 60)
    
    if render:
        logger.info("🎮 Rendering enabled - you'll see visual episodes!")
        logger.info("   Close pygame windows when done watching")
    
    try:
        # Initialize CUPID with flexible configuration
        logger.info(f"📋 Loading configuration: {config_name}")
        if config_name == "smoke_test":
            config = Config.smoke_test(max_episodes=max_episodes or 20)
        elif config_name == "quick_demo":
            if max_episodes is None:
                config = Config.quick_demo()  # Default: 1000 demos
            else:
                config = Config.for_demos(max_episodes)
        elif config_name == "default":
            config = Config.default(max_episodes=max_episodes)
        else:
            config = Config.for_demos(max_episodes or 1000)
        
        logger.info(f"✅ Configuration loaded: {config.dataset_name}")
        logger.info(f"   Device: {config.device}")
        logger.info(f"   Max episodes: {config.max_episodes or 'all available'}")
        
        # Initialize CUPID
        cupid = CUPID(config)
        
        # Enable rendering if requested
        if render:
            try:
                from cupid.evaluation import TaskEvaluator
                cupid.task_evaluator = TaskEvaluator(config, render_mode='human')
                logger.info("✅ Rendering enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to enable rendering: {e}")
                render = False
        
        logger.info(f"📊 Dataset: {config.dataset_name}")
        logger.info(f"🎯 Total demonstrations available: {len(cupid.dataset)}")
        logger.info(f"🎯 Selection ratio: {config.influence.selection_ratio*100:.0f}%")
        logger.info(f"🏋️ Training steps: {config.training.num_steps:,}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize CUPID: {e}")
        return 1
    
    try:
        # Step 1: Train baseline policy (only if needed)
        logger.info("📈 Step 1: Training baseline policy...")
        baseline_result = cupid.train_baseline()
        
        # Handle both cases: loaded existing policy or newly trained policy with loss history
        if isinstance(baseline_result, tuple):
            baseline_policy, baseline_loss_history = baseline_result
            logger.info("✅ Baseline policy trained and saved (with loss history)")
        else:
            baseline_policy = baseline_result
            baseline_loss_history = None  # No loss history for loaded policy
            logger.info("✅ Baseline policy loaded from checkpoint")
        
    except Exception as e:
        logger.error(f"❌ Failed to train baseline policy: {e}")
        return 1
    
    try:
        # Step 2: Compute influence scores for all demonstrations
        logger.info("🧠 Step 2: Computing influence scores...")
        influence_scores = cupid.compute_influence_scores(baseline_policy)
        logger.info(f"✅ Computed influence scores for {len(influence_scores)} demonstrations")
        
        # Show influence statistics
        influence_stats = cupid.get_influence_statistics(influence_scores)
        logger.info(f"   Influence score range: {influence_stats['min']:.4f} to {influence_stats['max']:.4f}")
        logger.info(f"   Mean: {influence_stats['mean']:.4f}, Std: {influence_stats['std']:.4f}")
        logger.info(f"   Median: {influence_stats['median']:.4f}, Q75: {influence_stats['q75']:.4f}")
        
        # Show some examples of high and low influence demonstrations
        sorted_indices = np.argsort(influence_scores)[::-1]
        logger.info(f"   Top 5 influence scores: {influence_scores[sorted_indices[:5]]}")
        logger.info(f"   Bottom 5 influence scores: {influence_scores[sorted_indices[-5:]]}")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute influence scores: {e}")
        return 1
    
    try:
        # Step 3: Select high-impact demonstrations using configured ratio
        logger.info("🎯 Step 3: Selecting high-impact demonstrations...")
        selected_indices = cupid.select_demonstrations(influence_scores)
        selection_ratio = len(selected_indices) / len(cupid.dataset)
        logger.info(f"✅ Selected {len(selected_indices)} demonstrations ({selection_ratio*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ Failed to select demonstrations: {e}")
        return 1
    
    try:
        # Step 4: Train curated policy and collect training history
        logger.info("🚀 Step 4: Training policy with curated data...")
        
        curated_policy, curated_loss_history = cupid.train_curated_policy(selected_indices)
        
        logger.info(f"✅ Curated policy trained with {len(selected_indices)} demonstrations")
        
        # Analyze curated training progress
        if curated_loss_history and len(curated_loss_history) > 100:
            initial_loss = np.mean(curated_loss_history[:50])
            final_loss = np.mean(curated_loss_history[-50:])
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            logger.info(f"   📈 Curated training progress: {initial_loss:.4f} → {final_loss:.4f} ({improvement:+.1f}% improvement)")
            
            if improvement < 5:
                logger.warning("⚠️ Limited training improvement - may need more steps or different hyperparameters")
        
    except Exception as e:
        logger.error(f"❌ Failed to train curated policy: {e}")
        return 1
    
    try:
        # Step 5: Task-based evaluation (the important part!)
        logger.info("🎯 Step 5: Task-based Performance Evaluation")
        logger.info("-" * 40)
        
        logger.info("   📊 Evaluating on actual task performance (success rate, rewards)...")
        
        # Compare policies (uses num_episodes from config by default)
        comparison_results = cupid.compare_policies(
            baseline_policy, curated_policy
        )
        
        # Show visual demonstrations if rendering enabled
        if render:
            logger.info("   🎥 Visual Policy Demonstrations:")
            rollout_count = 2 if config_name == "smoke_test" else 5
            logger.info(f"      Showing {rollout_count} rollouts of each policy...")
            
            # Demonstrate baseline policy
            cupid.task_evaluator.demonstrate_policy_rollouts(
                baseline_policy, "Baseline Policy", cupid.dataset, num_rollouts=rollout_count
            )
            
            # Demonstrate curated policy
            cupid.task_evaluator.demonstrate_policy_rollouts(
                curated_policy, "Curated Policy", cupid.dataset, num_rollouts=rollout_count
            )
        
    except Exception as e:
        logger.error(f"❌ Failed to evaluate policies: {e}")
        return 1
    
    try:
        # Step 6: Create visualization and print final report
        logger.info("📊 Step 6: Final Report & Visualization")
        logger.info("=" * 60)
        
        # Create visualization
        output_path = Path("outputs") / f"{config_name}_results.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_cupid_visualization(
            baseline_loss_history=baseline_loss_history,
            curated_loss_history=curated_loss_history,
            influence_scores=influence_scores,
            comparison_results=comparison_results,
            config=asdict(config),
            output_path=str(output_path)
        )
        logger.info(f"✅ Saved visualization to {output_path}")
        
        # Print final report
        _print_final_report(comparison_results, len(selected_indices), len(cupid.dataset))

    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
        return 1
    
    logger.info("✅ CUPID workflow completed successfully!")
    return 0


def _print_final_report(results: dict, num_curated: int, num_total: int):
    """Print final comparison report in a clean table."""
    
    baseline = results.get('baseline', {})
    curated = results.get('curated', {})
    improvements = results.get('improvements', {})
    
    # Header
    print("\n" + "="*70)
    print(" " * 25 + "CUPID Final Report")
    print("="*70)
    
    # Summary
    print(f"Data Curated: {num_curated}/{num_total} demonstrations ({num_curated/num_total:.1%})")
    print("-"*70)
    
    # Table header
    print(f"{'Metric':<25} {'Baseline':<15} {'Curated':<15} {'Improvement':<15}")
    print("-"*70)
    
    # Metrics to display
    metric_map = {
        'success_rate': 'Success Rate',
        'avg_reward': 'Avg. Reward',
        'avg_final_distance': 'Final Distance',
        'action_consistency': 'Action Consistency'
    }
    
    for key, name in metric_map.items():
        baseline_val = baseline.get(key, 0)
        curated_val = curated.get(key, 0)
        improvement = improvements.get(f"{key}_improvement", 0)
        
        # Original, clean formatting
        if 'rate' in key:
            baseline_str = f"{baseline_val:.1%}"
            curated_str = f"{curated_val:.1%}"
        elif 'distance' in key:
            baseline_str = f"{baseline_val:.2f}"
            curated_str = f"{curated_val:.2f}"
        else:
            baseline_str = f"{baseline_val:.3f}"
            curated_str = f"{curated_val:.3f}"
        
        improvement_str = f"{improvement:+.1f}%" if isinstance(improvement, (int, float)) else "N/A"
        
        print(f"{name:<25} {baseline_str:<15} {curated_str:<15} {improvement_str:<15}")
        
    print("="*70)
    logger.info("Final report generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUPID Workflow Example")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to use")
    parser.add_argument("--config", dest="config_name", type=str, default="quick_demo",
                        help="Configuration preset (smoke_test, quick_demo, default)")
    
    args = parser.parse_args()
    
    sys.exit(main(
        render=args.render,
        max_episodes=args.max_episodes,
        config_name=args.config_name
    )) 