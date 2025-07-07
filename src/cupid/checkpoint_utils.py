"""
Checkpoint utilities for CUPID.

This module provides utilities for managing and inspecting checkpoint files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from datetime import datetime

logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, List[Path]]:
    """
    List all checkpoint files organized by type.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Dictionary with checkpoint types as keys and lists of paths as values
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return {}
    
    checkpoints = {
        'baseline': [],
        'curated': [],
        'other': []
    }
    
    for checkpoint_file in checkpoint_path.glob("*.pth"):
        if checkpoint_file.name.startswith("baseline_"):
            checkpoints['baseline'].append(checkpoint_file)
        elif checkpoint_file.name.startswith("curated_"):
            checkpoints['curated'].append(checkpoint_file)
        else:
            checkpoints['other'].append(checkpoint_file)
    
    # Sort by modification time (newest first)
    for key in checkpoints:
        checkpoints[key].sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoints


def inspect_checkpoint(filepath: Path) -> Dict[str, Any]:
    """
    Inspect a checkpoint file and return its metadata.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        info = {
            'filepath': str(filepath),
            'filename': filepath.name,
            'file_size_mb': filepath.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            'has_metadata': 'metadata' in checkpoint,
            'has_config': 'config' in checkpoint,
            'has_model': 'model_state_dict' in checkpoint,
        }
        
        # Add metadata if available
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            info.update({
                'policy_type': metadata.get('policy_type', 'unknown'),
                'num_trajectories': metadata.get('num_trajectories', '?'),
                'num_steps': metadata.get('num_steps', '?'),
                'dataset_name': metadata.get('dataset_name', 'unknown'),
                'selection_ratio': metadata.get('selection_ratio', '?'),
                'config_name': metadata.get('config_name', 'unknown'),
                'saved_at': metadata.get('saved_at', 'unknown'),
                'training_steps': metadata.get('training_steps', '?'),
            })
            
            # Add curated-specific info
            if metadata.get('policy_type') == 'curated':
                info['total_available_trajectories'] = metadata.get('total_available_trajectories', '?')
                info['selected_indices_count'] = len(metadata.get('selected_indices', []))
        
        return info
        
    except Exception as e:
        return {
            'filepath': str(filepath),
            'filename': filepath.name,
            'error': str(e),
            'file_size_mb': filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0,
        }


def print_checkpoint_summary(checkpoint_dir: str = "checkpoints") -> None:
    """
    Print a summary of all checkpoints in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not any(checkpoints.values()):
        logger.info(f"📁 No checkpoints found in {checkpoint_dir}")
        return
    
    logger.info(f"📁 Checkpoint Summary: {checkpoint_dir}")
    logger.info("=" * 60)
    
    for checkpoint_type, files in checkpoints.items():
        if not files:
            continue
            
        logger.info(f"\n🔹 {checkpoint_type.upper()} POLICIES ({len(files)} found):")
        
        for filepath in files:
            info = inspect_checkpoint(filepath)
            
            if 'error' in info:
                logger.error(f"   {info['filename']} - Error: {info['error']}")
                continue
            
            # Format the display
            size_str = f"{info['file_size_mb']:.1f}MB"
            modified_str = info['modified'][:19].replace('T', ' ')  # Remove microseconds and T
            
            logger.info(f"   {info['filename']}")
            logger.info(f"      Dataset: {info.get('num_trajectories', '?')} trajectories, {info.get('num_steps', '?')} steps")
            logger.info(f"      Config: {info.get('config_name', 'unknown')}")
            logger.info(f"      Modified: {modified_str} ({size_str})")
            
            if info.get('policy_type') == 'curated':
                ratio = info.get('selection_ratio', 0)
                if isinstance(ratio, (int, float)) and ratio > 0:
                    logger.info(f"      Selection: {ratio:.1%} of {info.get('total_available_trajectories', '?')} trajectories")


def cleanup_old_checkpoints(checkpoint_dir: str = "checkpoints", keep_latest: int = 3) -> None:
    """
    Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_latest: Number of latest checkpoints to keep per type
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    removed_count = 0
    
    for checkpoint_type, files in checkpoints.items():
        if len(files) > keep_latest:
            files_to_remove = files[keep_latest:]  # Already sorted by modification time
            
            for filepath in files_to_remove:
                try:
                    filepath.unlink()
                    logger.info(f"Removed old checkpoint: {filepath.name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {filepath.name}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old checkpoint(s)")
    else:
        logger.info("No cleanup needed - all checkpoints are recent")


def find_matching_baseline(checkpoint_dir: str, num_trajectories: int, num_steps: int) -> Optional[Path]:
    """
    Find a baseline checkpoint that matches the given dataset parameters.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        num_trajectories: Number of trajectories in dataset
        num_steps: Number of steps in dataset
        
    Returns:
        Path to matching baseline checkpoint, or None if not found
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    for baseline_path in checkpoints['baseline']:
        info = inspect_checkpoint(baseline_path)
        
        if (info.get('num_trajectories') == num_trajectories and 
            info.get('num_steps') == num_steps):
            return baseline_path
    
    return None 