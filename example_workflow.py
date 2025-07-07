#!/usr/bin/env python3
"""
CUPID Example: Complete workflow demonstration

This example shows how to use CUPID for robot imitation learning data curation.
Based on the paper "CUPID: Curating Data your Robot Loves with Influence Functions"
which demonstrates that training with ~25-33% of curated data can achieve state-of-the-art performance.

IMPORTANT FIXES APPLIED:
- ✅ Training steps increased from 10 to 5000+ (matching LeRobot standards)
- ✅ Action clipping added to prevent out-of-bounds actions
- ✅ Environment consistency fixed for influence computation
- ✅ Device compatibility issues resolved
"""

import torch
from pathlib import Path
import numpy as np
import logging
import sys
import argparse
from typing import Optional
from dataclasses import asdict
import time

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

# Add src and lerobot to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
# HACK: Forcefully add the lerobot path to resolve import issues
lerobot_path = "/home/mphielipp/robotsw/lerobot"
if lerobot_path not in sys.path:
    sys.path.insert(0, lerobot_path)

try:
    from cupid import CUPID, Config
    from cupid.visualization import create_cupid_visualization
except ImportError as e:
    logger.error(f"Failed to import CUPID modules: {e}")
    logger.error("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)


def setup_error_handling():
    """Setup comprehensive error handling for the workflow."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Workflow interrupted by user")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error("💥 Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception


def validate_environment():
    """Validate that the environment is properly set up."""
    try:
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
            
        # Check if we can import all required components
        from src.cupid.cupid import CUPID
        from src.cupid.config import Config
        logger.info("All CUPID components imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def main(render=False, max_demonstrations=None, config_name="quick_demo", selection_ratio=None, force_retrain=False, environment="cupid", lerobot_path=None, generate_videos=False):
    """
    Complete CUPID workflow example with comprehensive error handling.
    
    Args:
        render: Enable visual rendering
        max_demonstrations: Maximum number of demonstrations to use
        config_name: Configuration preset name
        selection_ratio: Override selection ratio (e.g., 0.33 for 33%)
        force_retrain: Force retraining even if checkpoints exist
    """
    
    # Setup error handling
    setup_error_handling()
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed - exiting")
        return 1
    
    logger.info("CUPID: Curating Data your Robot Loves with Influence Functions")
    logger.info("=" * 60)
    
    if render:
        logger.info("🎮 Rendering enabled - you'll see visual episodes!")
        logger.info("   Close pygame windows when done watching")
    
    try:
        # Initialize CUPID with flexible configuration
        logger.info(f"Loading configuration: {config_name}")
        if config_name == "smoke_test":
            config = Config.smoke_test(max_demonstrations=max_demonstrations or 20)
        elif config_name == "micro_test":
            config = Config.micro_test(max_demonstrations=max_demonstrations or 10)
        elif config_name == "quick_demo":
            if max_demonstrations is None:
                config = Config.quick_demo()  # Default: 1000 demos
            else:
                config = Config.for_demos(max_demonstrations)
        elif config_name == "default":
            config = Config.default(max_demonstrations=max_demonstrations)
        else:
            config = Config.for_demos(max_demonstrations or 1000)
        
        # Override config with CLI args
        if selection_ratio is not None:
            config.influence.selection_ratio = selection_ratio
        if force_retrain:
            config.force_retrain = True
        if environment:
            config.environment_type = environment
        if lerobot_path:
            config.lerobot_path = lerobot_path
        
        logger.info(f"Configuration loaded: {config.dataset_name}")
        logger.info(f"   Device: {config.device}")
        logger.info(f"   Environment: {config.environment_type}")
        logger.info(f"   Max demonstrations: {config.max_demonstrations or 'all available'}")
        logger.info(f"   Selection ratio: {config.influence.selection_ratio*100:.0f}%")
        logger.info(f"   Force retrain: {config.force_retrain}")
        
        # Initialize CUPID without rendering (rendering only for final demonstrations)
        cupid = CUPID(config, render_mode=None)
        
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Total demonstrations available: {len(cupid.dataset)}")
        logger.info(f"Selection ratio: {config.influence.selection_ratio*100:.0f}%")
        logger.info(f"Training steps: {config.training.num_steps:,}")
        
    except Exception as e:
        logger.error(f"Failed to initialize CUPID: {e}")
        return 1
    
    try:
        # Step 1: Train baseline policy (only if needed)
        logger.info("Step 1: Training baseline policy...")
        baseline_result = cupid.train_baseline()
        
        # Handle both cases: loaded existing policy or newly trained policy with loss history
        if isinstance(baseline_result, tuple):
            baseline_policy, baseline_loss_history = baseline_result
            logger.info("Baseline policy trained and saved (with loss history)")
        else:
            baseline_policy = baseline_result
            baseline_loss_history = None  # No loss history for loaded policy
            logger.info("Baseline policy loaded from checkpoint")
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to train baseline policy: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1
    
    try:
        # Step 2: Compute influence scores for all demonstrations
        logger.info("Step 2: Computing influence scores...")
        influence_scores = cupid.compute_influence_scores(baseline_policy)
        logger.info(f"Computed influence scores for {len(influence_scores)} demonstrations")
        
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
        logger.error(f"Failed to compute influence scores: {e}")
        return 1
    
    try:
        # Step 3: Select high-impact demonstrations using configured ratio
        logger.info("Step 3: Selecting high-impact demonstrations...")
        selected_indices = cupid.select_demonstrations(influence_scores)
        selection_ratio = len(selected_indices) / len(cupid.dataset)
        logger.info(f"Selected {len(selected_indices)} demonstrations ({selection_ratio*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to select demonstrations: {e}")
        return 1
    
    try:
        # Step 4: Train curated policy and collect training history
        logger.info("🚀 Step 4: Training policy with curated data...")
        
        curated_policy, curated_loss_history = cupid.train_curated_policy(selected_indices)
        
        logger.info(f"Curated policy trained with {len(selected_indices)} demonstrations")
        
        # Analyze curated training progress
        if curated_loss_history and len(curated_loss_history) > 100:
            initial_loss = np.mean(curated_loss_history[:50])
            final_loss = np.mean(curated_loss_history[-50:])
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            logger.info(f"   Curated training progress: {initial_loss:.4f} → {final_loss:.4f} ({improvement:+.1f}% improvement)")
            
            if improvement < 5:
                logger.warning("Limited training improvement - may need more steps or different hyperparameters")
        
    except Exception as e:
        logger.error(f"Failed to train curated policy: {e}")
        return 1
    
    try:
        # Step 5: Task-based evaluation (the important part!)
        logger.info("Step 5: Task-based Performance Evaluation")
        logger.info("-" * 40)
        
        logger.info("   Evaluating on actual task performance (success rate, rewards)...")
        
        # Compare policies (uses num_episodes from config by default)
        comparison_results = cupid.compare_policies(
            baseline_policy, curated_policy
        )
        
        # Show visual demonstrations if rendering enabled
        if render:
            logger.info("   🎥 Visual Policy Demonstrations:")
            rollout_count = 2 if config_name == "smoke_test" else 5
            logger.info(f"      Showing {rollout_count} rollouts of each policy...")
            
            # Create a separate evaluator with rendering enabled for demonstrations only
            from src.cupid.evaluation import TaskEvaluator
            demo_evaluator = TaskEvaluator(config, render_mode='human')
            
            # Flatten dataset for demonstrations (evaluator expects individual steps)
            flat_dataset = []
            for trajectory in cupid.dataset:
                flat_dataset.extend(trajectory)
            
            # Demonstrate baseline policy
            demo_evaluator.demonstrate_policy_rollouts(
                baseline_policy, "Baseline Policy", flat_dataset, num_rollouts=rollout_count
            )
            
            # Demonstrate curated policy
            demo_evaluator.demonstrate_policy_rollouts(
                curated_policy, "Curated Policy", flat_dataset, num_rollouts=rollout_count
            )
        
        # Generate videos only if explicitly requested
        if generate_videos:
            logger.info("   Generating video comparisons...")
            video_output_dir = Path("outputs") / "videos"
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create evaluator for video generation (no interactive rendering)
            from src.cupid.evaluation import TaskEvaluator
            video_evaluator = TaskEvaluator(config, render_mode=None)
            
            # Flatten dataset for video generation
            flat_dataset = []
            for trajectory in cupid.dataset:
                flat_dataset.extend(trajectory)
            
            # Generate videos for both policies
            video_count = 3 if config_name == "smoke_test" else 5
            logger.info(f"      Generating {video_count} videos for each policy...")
            
            baseline_videos = video_evaluator.generate_policy_videos(
                baseline_policy, "Baseline_Policy", flat_dataset, 
                num_videos=video_count, output_dir=str(video_output_dir)
            )
            
            curated_videos = video_evaluator.generate_policy_videos(
                curated_policy, "Curated_Policy", flat_dataset, 
                num_videos=video_count, output_dir=str(video_output_dir)
            )
            
            all_videos = baseline_videos + curated_videos
            if all_videos:
                logger.info(f"Generated {len(all_videos)} video files:")
                for video_path in all_videos:
                    logger.info(f"   {video_path}")
            else:
                logger.info("   Video generation skipped (requires additional dependencies)")
        else:
            logger.info("   Video generation skipped (use --generate-videos to enable)")
        
    except Exception as e:
        logger.error(f"Failed to evaluate policies: {e}")
        return 1
    
    try:
        # Step 6: Create visualization and print final report
        logger.info("Step 6: Final Report & Visualization")
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
        logger.info(f"Saved visualization to {output_path}")
        
        # Print final report
        _print_final_report(comparison_results, len(selected_indices), len(cupid.dataset))

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return 1
    
    logger.info("CUPID workflow completed successfully!")
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
        'avg_final_distance': 'Final Distance'
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CUPID workflow")
    parser.add_argument("--render", action="store_true", help="Show visual demonstrations")
    parser.add_argument("--max-demonstrations", type=int, help="Maximum number of demonstrations to use")
    parser.add_argument("--config", default="quick_demo", choices=["smoke_test", "micro_test", "quick_demo", "for_demos", "default"], help="Configuration to use")
    parser.add_argument("--selection-ratio", type=float, help="Selection ratio (e.g., 0.25 for 25%)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if checkpoints exist")
    parser.add_argument("--environment", default="cupid", choices=["cupid", "lerobot"], help="Environment type")
    parser.add_argument("--lerobot-path", help="Path to LeRobot installation")
    parser.add_argument("--generate-videos", action="store_true", help="Generate demonstration videos (lerobot only)")
    
    args = parser.parse_args()
    
    results = main(
        render=args.render,
        max_demonstrations=args.max_demonstrations,
        config_name=args.config,
        selection_ratio=args.selection_ratio,
        force_retrain=args.force_retrain,
        environment=args.environment,
        lerobot_path=args.lerobot_path,
        generate_videos=args.generate_videos
    ) 