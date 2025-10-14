#!/usr/bin/env python3
"""
Monitor training progress for S2Net
"""

import os
import time
import subprocess
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def check_training_status():
    """Check if training is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_latest_logs():
    """Get the latest training logs."""
    log_dir = 'logs/kitti'
    if not os.path.exists(log_dir):
        return "No logs found"
    
    # Find the latest log file
    log_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not log_files:
        return "No tensorboard logs found"
    
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(log_dir, latest_log)
    
    try:
        # Read tensorboard logs
        ea = EventAccumulator(log_path)
        ea.Reload()
        
        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        logs = "Training Logs:\n"
        logs += "=" * 50 + "\n"
        
        for tag in scalar_tags:
            if 'Train' in tag or 'Val' in tag:
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    latest_value = scalar_events[-1].value
                    step = scalar_events[-1].step
                    logs += f"{tag}: {latest_value:.6f} (step {step})\n"
        
        return logs
    except Exception as e:
        return f"Error reading logs: {e}"

def plot_training_curves():
    """Plot training curves from tensorboard logs."""
    log_dir = 'logs/kitti'
    if not os.path.exists(log_dir):
        print("No logs directory found")
        return
    
    try:
        # Find the latest log file
        log_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
        if not log_files:
            print("No tensorboard logs found")
            return
        
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join(log_dir, latest_log)
        
        # Read tensorboard logs
        ea = EventAccumulator(log_path)
        ea.Reload()
        
        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        plot_idx = 0
        for tag in scalar_tags:
            if plot_idx >= 4:
                break
                
            if 'Train' in tag or 'Val' in tag:
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    
                    axes[plot_idx].plot(steps, values)
                    axes[plot_idx].set_title(tag)
                    axes[plot_idx].set_xlabel('Step')
                    axes[plot_idx].set_ylabel('Value')
                    axes[plot_idx].grid(True)
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Training curves saved to training_curves.png")
        
    except Exception as e:
        print(f"Error plotting curves: {e}")

def check_checkpoints():
    """Check for available checkpoints."""
    checkpoint_dir = 'checkpoints/kitti'
    if not os.path.exists(checkpoint_dir):
        return "No checkpoints directory found"
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return "No checkpoints found"
    
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]
    
    # Get file size and modification time
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    stat = os.stat(checkpoint_path)
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = time.ctime(stat.st_mtime)
    
    return f"Latest checkpoint: {latest_checkpoint}\nSize: {size_mb:.1f} MB\nModified: {mod_time}"

def main():
    """Main monitoring function."""
    print("S2Net Training Monitor")
    print("=" * 50)
    
    # Check if training is running
    if check_training_status():
        print("✓ Training is running")
    else:
        print("✗ Training is not running")
    
    print("\n" + "=" * 50)
    
    # Get latest logs
    logs = get_latest_logs()
    print(logs)
    
    print("\n" + "=" * 50)
    
    # Check checkpoints
    checkpoints = check_checkpoints()
    print("Checkpoints:")
    print(checkpoints)
    
    print("\n" + "=" * 50)
    
    # Plot training curves
    print("Generating training curves...")
    plot_training_curves()

if __name__ == '__main__':
    main()
