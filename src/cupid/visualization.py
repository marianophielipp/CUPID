"""
Visualization module for CUPID results.

This module provides clean, meaningful visualizations for CUPID analysis including
training progress, influence scores, demonstration selection, and performance comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set up clean plot style
plt.style.use('default')
sns.set_palette("husl")


class CUPIDVisualizer:
    """Creates clean visualizations for CUPID analysis results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'baseline': '#2E86AB',      # Blue
            'curated': '#A23B72',       # Purple
            'selected': '#F18F01',      # Orange
            'unselected': '#C73E1D',    # Red
            'background': '#F5F5F5',    # Light gray
            'text': '#2C3E50'           # Dark blue-gray
        }
    
    def create_comprehensive_report(
        self,
        influence_scores: np.ndarray,
        selected_indices: List[int],
        baseline_loss_history: List[float],
        curated_loss_history: List[float],
        baseline_metrics: Dict[str, float],
        curated_metrics: Dict[str, float],
        config: Dict,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive visualization report of CUPID results.
        
        Args:
            influence_scores: Array of influence scores for all demonstrations
            selected_indices: Indices of selected demonstrations
            baseline_loss_history: Training loss history for baseline policy
            curated_loss_history: Training loss history for curated policy
            baseline_metrics: Evaluation metrics for baseline policy
            curated_metrics: Evaluation metrics for curated policy
            config: Configuration dictionary with training details
            save_path: Path to save the visualization (optional)
        """
        # Create figure with subplots - increased figure height and adjusted spacing
        fig = plt.figure(figsize=(16, 14))  # Increased height from 12 to 14
        fig.suptitle('CUPID: Data Curation Analysis Report', 
                    fontsize=20, fontweight='bold', y=0.96)  # Moved title up slightly
        
        # Create grid layout with better spacing
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], width_ratios=[1, 1, 1],
                             hspace=0.4, wspace=0.3, top=0.92, bottom=0.08)  # Increased hspace and adjusted margins
        
        # 1. Training Progress Comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_training_progress(ax1, baseline_loss_history, curated_loss_history)
        
        # 2. Influence Score Distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_influence_distribution(ax2, influence_scores, selected_indices)
        
        # 3. Selection Overview (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_selection_overview(ax3, len(influence_scores), len(selected_indices))
        
        # 4. Performance Comparison (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_performance_comparison(ax4, baseline_metrics, curated_metrics)
        
        # 5. Influence Score Ranking (middle center & right - spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_influence_ranking(ax5, influence_scores, selected_indices)
        
        # 6. Summary Statistics (bottom - spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_summary_stats(ax6, influence_scores, selected_indices, 
                               baseline_metrics, curated_metrics, config)
        
        # Add subtle background
        fig.patch.set_facecolor(self.colors['background'])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'], pad_inches=0.2)  # Added padding
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_training_progress(self, ax, baseline_history: List[float], 
                               curated_history: List[float]) -> None:
        """Plot training loss progression for both policies."""
        
        # Handle case where baseline was loaded from checkpoint (no training history)
        if baseline_history is not None and len(baseline_history) > 0:
            ax.plot(baseline_history, color=self.colors['baseline'], 
                    linewidth=2, label=f'Baseline ({len(baseline_history)} steps)', alpha=0.8)
            has_baseline = True
        else:
            has_baseline = False
        
        # Always plot curated history (should always be available)
        if curated_history is not None and len(curated_history) > 0:
            ax.plot(curated_history, color=self.colors['curated'], 
                    linewidth=2, label=f'Curated ({len(curated_history)} steps)', alpha=0.8)
            has_curated = True
        else:
            has_curated = False
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        
        # Update title and add info about missing baseline
        if has_baseline and has_curated:
            ax.set_title('Training Progress Comparison', fontweight='bold')
        elif has_curated and not has_baseline:
            ax.set_title('Training Progress (Baseline loaded from checkpoint)', fontweight='bold')
            # Add text explaining why baseline is missing
            ax.text(0.02, 0.98, 'Baseline: Loaded from existing checkpoint\n(no training history available)', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                   fontsize=9)
        elif has_baseline and not has_curated:
            ax.set_title('Training Progress (Curated training failed)', fontweight='bold')
        else:
            ax.set_title('Training Progress (No data available)', fontweight='bold')
            ax.text(0.5, 0.5, 'No training history available\n(Both policies loaded from checkpoints)', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
                   fontsize=10)
        
        if has_baseline or has_curated:
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    def _plot_influence_distribution(self, ax, influence_scores: np.ndarray, 
                                   selected_indices: List[int]) -> None:
        """Plot distribution of influence scores with selection threshold."""
        # Create histogram
        ax.hist(influence_scores, bins=30, alpha=0.7, color=self.colors['baseline'], 
                edgecolor='white', linewidth=0.5)
        
        # Mark selection threshold
        if selected_indices:
            threshold = np.sort(influence_scores)[-len(selected_indices)]
            ax.axvline(threshold, color=self.colors['selected'], linestyle='--', 
                      linewidth=2, label=f'Selection Threshold')
            ax.legend()
        
        ax.set_xlabel('Influence Score')
        ax.set_ylabel('Number of Demonstrations')
        ax.set_title('Influence Score Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_selection_overview(self, ax, total_demos: int, selected_demos: int) -> None:
        """Plot pie chart showing selection overview."""
        unselected = total_demos - selected_demos
        sizes = [selected_demos, unselected]
        labels = [f'Selected\n({selected_demos})', f'Unselected\n({unselected})']
        colors = [self.colors['selected'], self.colors['unselected']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 10})
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax.set_title('Demonstration Selection', fontweight='bold')
    
    def _plot_performance_comparison(self, ax, baseline_metrics: Dict[str, float], 
                                   curated_metrics: Dict[str, float]) -> None:
        """Plot performance comparison between baseline and curated policies."""
        # Always show the main task metrics
        metrics = ['success_rate', 'avg_reward', 'avg_final_distance']
        metric_labels = ['Success Rate', 'Avg Reward', 'Final Distance']
        
        baseline_values = [baseline_metrics.get(m, 0) for m in metrics]
        curated_values = [curated_metrics.get(m, 0) for m in metrics]
        
        # For distance, invert so lower is better (show improvement correctly)
        if 'avg_final_distance' in metrics:
            dist_idx = metrics.index('avg_final_distance')
            # Normalize distance to 0-1 scale and invert (1 - normalized_distance)
            max_dist = max(baseline_values[dist_idx], curated_values[dist_idx], 1.0)
            if max_dist > 0:
                baseline_values[dist_idx] = 1.0 - (baseline_values[dist_idx] / max_dist)
                curated_values[dist_idx] = 1.0 - (curated_values[dist_idx] / max_dist)
                metric_labels[dist_idx] = 'Distance Quality\n(1 - normalized)'
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', 
                      color=self.colors['baseline'], alpha=0.8)
        bars2 = ax.bar(x + width/2, curated_values, width, label='Curated', 
                      color=self.colors['curated'], alpha=0.8)
        
        ax.set_xlabel('Task Performance Metrics')
        ax.set_ylabel('Performance Score')
        ax.set_title('Baseline vs Curated Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=0, ha='center')  # Better readability
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars with better formatting
        for bars, name in [(bars1, 'Baseline'), (bars2, 'Curated')]:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                # Format values appropriately
                if metrics[i] == 'success_rate':
                    label = f'{height:.1%}'
                elif metrics[i] == 'avg_reward':
                    label = f'{height:.3f}'
                else:
                    label = f'{height:.2f}'
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add a text box showing the comparison clearly
        baseline_reward = baseline_metrics.get('avg_reward', 0)
        curated_reward = curated_metrics.get('avg_reward', 0)
        baseline_success = baseline_metrics.get('success_rate', 0)
        curated_success = curated_metrics.get('success_rate', 0)
        
        comparison_text = f"""COMPARISON SUMMARY:
Baseline: {baseline_success:.1%} success, {baseline_reward:.3f} reward
Curated:  {curated_success:.1%} success, {curated_reward:.3f} reward"""
        
        ax.text(0.02, 0.98, comparison_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8, fontfamily='monospace')
    
    def _plot_influence_ranking(self, ax, influence_scores: np.ndarray, 
                              selected_indices: List[int]) -> None:
        """Plot influence scores ranked from highest to lowest."""
        # Sort scores and get ranking
        sorted_indices = np.argsort(influence_scores)[::-1]
        sorted_scores = influence_scores[sorted_indices]
        
        # Create color array (selected vs unselected)
        colors = ['orange' if i in selected_indices else 'lightblue' 
                 for i in sorted_indices]
        
        # Plot ranking
        ax.bar(range(len(sorted_scores)), sorted_scores, color=colors, alpha=0.7)
        
        # Mark selection cutoff
        if selected_indices:
            cutoff = len(selected_indices)
            ax.axvline(cutoff - 0.5, color='red', linestyle='--', linewidth=2, 
                      label=f'Selection Cutoff (Top {cutoff})')
            ax.legend()
        
        ax.set_xlabel('Demonstration Rank')
        ax.set_ylabel('Influence Score')
        ax.set_title('Influence Score Ranking (Highest to Lowest)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, ax, influence_scores: np.ndarray, 
                          selected_indices: List[int],
                          baseline_metrics: Dict[str, float],
                          curated_metrics: Dict[str, float],
                          config: Dict) -> None:
        """Plot summary statistics table."""
        ax.axis('off')  # Turn off axis
        
        # Calculate statistics
        total_demos = len(influence_scores)
        selected_demos = len(selected_indices)
        selection_ratio = selected_demos / total_demos if total_demos > 0 else 0
        
        baseline_success = baseline_metrics.get('success_rate', 'N/A')
        curated_success = curated_metrics.get('success_rate', 'N/A')
        
        # Robust formatting helper
        def format_val(val, spec):
            if isinstance(val, (int, float)):
                return f"{val:{spec}}"
            return str(val)
            
        # Calculate reward improvement for better summary
        baseline_reward = baseline_metrics.get('avg_reward', 0)
        curated_reward = curated_metrics.get('avg_reward', 0)
        
        if baseline_reward != 0:
            reward_improvement = ((curated_reward - baseline_reward) / abs(baseline_reward)) * 100
        else:
            reward_improvement = curated_reward * 100 if curated_reward != 0 else 0
        
        # Table data with more relevant metrics
        table_data = [
            ['Configuration', f"{config.get('dataset_name', 'Unknown')[:20]}..."],
            ['Data Selection', f"{selected_demos}/{total_demos} ({selection_ratio:.1%})"],
            ['Baseline Reward', format_val(baseline_reward, ".3f")],
            ['Curated Reward', format_val(curated_reward, ".3f")],
            ['Reward Improvement', f"{reward_improvement:+.1f}%"],
            ['Success Rate', f"{format_val(baseline_success, '.1%')} → {format_val(curated_success, '.1%')}"]
        ]
        
        # Create table
        table = ax.table(cellText=table_data, 
                         colLabels=['Statistic', 'Value'],
                         cellLoc='left', 
                         loc='center',
                         colWidths=[0.3, 0.7])
                         
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust vertical scaling
        
        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(self.colors['text'])
            else:
                cell.set_facecolor(self.colors['background'])
            
            if col == 0:
                cell.set_text_props(weight='bold')

        # Position title with more space from table
        ax.set_title('CUPID Results Summary', fontweight='bold', pad=15)
        
        ax.text(0.01, 0.05, f"Config: {config.get('dataset_name')} / {config.get('max_episodes', 'N/A')} demos", 
               ha='left', va='bottom', fontsize=8, alpha=0.7)


def create_cupid_visualization(
    influence_scores: np.ndarray,
    baseline_loss_history: List[float],
    curated_loss_history: List[float],
    comparison_results: Dict[str, Dict[str, float]],
    config: Dict,
    output_path: Optional[str] = None,
):
    """
    Creates a comprehensive visualization report for CUPID results.

    This function is a convenient wrapper around the CUPIDVisualizer class.

    Args:
        influence_scores: Array of influence scores.
        baseline_loss_history: Training loss history for the baseline policy.
        curated_loss_history: Training loss history for the curated policy.
        comparison_results: Dictionary containing 'baseline' and 'curated' metrics.
        config: The main configuration dictionary.
        output_path: Path to save the visualization.
    """
    visualizer = CUPIDVisualizer()
    
    # Extract metrics from the comparison dictionary
    baseline_metrics = comparison_results.get('baseline', {})
    curated_metrics = comparison_results.get('curated', {})
    
    # Determine selected indices from influence scores and config
    selection_ratio = config.get('influence', {}).get('selection_ratio', 0.33)
    num_to_select = int(len(influence_scores) * selection_ratio)
    selected_indices = np.argsort(influence_scores)[-num_to_select:].tolist()
    
    visualizer.create_comprehensive_report(
        influence_scores=influence_scores,
        selected_indices=selected_indices,
        baseline_loss_history=baseline_loss_history or [],
        curated_loss_history=curated_loss_history or [],
        baseline_metrics=baseline_metrics,
        curated_metrics=curated_metrics,
        config=config,
        save_path=output_path
    ) 