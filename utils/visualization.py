import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def save_point_cloud_sequence(points: torch.Tensor, 
                            filename: str,
                            format: str = 'ply') -> None:
    """
    Save point cloud sequence to file.
    
    Args:
        points: Point cloud sequence [seq_len, num_points, 3] or [batch_size, seq_len, num_points, 3]
        filename: Output filename
        format: File format ('ply', 'xyz', 'txt')
    """
    if points.dim() == 4:
        points = points[0]  # Take first batch
    
    points_np = points.cpu().numpy()
    seq_len, num_points, _ = points_np.shape
    
    if format.lower() == 'ply':
        save_ply_sequence(points_np, filename)
    elif format.lower() == 'xyz':
        save_xyz_sequence(points_np, filename)
    elif format.lower() == 'txt':
        save_txt_sequence(points_np, filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_ply_sequence(points: np.ndarray, filename: str) -> None:
    """Save point cloud sequence as PLY files."""
    seq_len, num_points, _ = points.shape
    
    for t in range(seq_len):
        ply_filename = filename.replace('.ply', f'_t{t:03d}.ply')
        
        with open(ply_filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {num_points}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            
            for i in range(num_points):
                f.write(f'{points[t, i, 0]:.6f} {points[t, i, 1]:.6f} {points[t, i, 2]:.6f}\n')


def save_xyz_sequence(points: np.ndarray, filename: str) -> None:
    """Save point cloud sequence as XYZ files."""
    seq_len, num_points, _ = points.shape
    
    for t in range(seq_len):
        xyz_filename = filename.replace('.xyz', f'_t{t:03d}.xyz')
        
        with open(xyz_filename, 'w') as f:
            for i in range(num_points):
                f.write(f'{points[t, i, 0]:.6f} {points[t, i, 1]:.6f} {points[t, i, 2]:.6f}\n')


def save_txt_sequence(points: np.ndarray, filename: str) -> None:
    """Save point cloud sequence as TXT files."""
    seq_len, num_points, _ = points.shape
    
    for t in range(seq_len):
        txt_filename = filename.replace('.txt', f'_t{t:03d}.txt')
        
        np.savetxt(txt_filename, points[t], fmt='%.6f')


def visualize_sequence(pred_points: torch.Tensor, 
                      target_points: torch.Tensor,
                      save_path: str = None,
                      show: bool = True,
                      max_timesteps: int = 5) -> None:
    """
    Visualize predicted and target point cloud sequences.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3] or [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [seq_len, num_points, 3] or [batch_size, seq_len, num_points, 3]
        save_path: Path to save visualization
        show: Whether to show the plot
        max_timesteps: Maximum number of timesteps to visualize
    """
    if pred_points.dim() == 4:
        pred_points = pred_points[0]  # Take first batch
    if target_points.dim() == 4:
        target_points = target_points[0]  # Take first batch
    
    pred_np = pred_points.cpu().numpy()
    target_np = target_points.cpu().numpy()
    
    seq_len = min(pred_np.shape[0], max_timesteps)
    
    # Create subplots
    fig = plt.figure(figsize=(15, 3 * seq_len))
    
    for t in range(seq_len):
        # Predicted points
        ax1 = fig.add_subplot(seq_len, 2, 2*t + 1, projection='3d')
        ax1.scatter(pred_np[t, :, 0], pred_np[t, :, 1], pred_np[t, :, 2], 
                   c='red', s=1, alpha=0.6)
        ax1.set_title(f'Predicted t={t}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Target points
        ax2 = fig.add_subplot(seq_len, 2, 2*t + 2, projection='3d')
        ax2.scatter(target_np[t, :, 0], target_np[t, :, 1], target_np[t, :, 2], 
                   c='blue', s=1, alpha=0.6)
        ax2.set_title(f'Target t={t}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_3d_interactive(pred_points: torch.Tensor, 
                            target_points: torch.Tensor,
                            save_path: str = None,
                            max_timesteps: int = 5) -> None:
    """
    Create interactive 3D visualization using Plotly.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3] or [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [seq_len, num_points, 3] or [batch_size, seq_len, num_points, 3]
        save_path: Path to save HTML file
        max_timesteps: Maximum number of timesteps to visualize
    """
    if pred_points.dim() == 4:
        pred_points = pred_points[0]  # Take first batch
    if target_points.dim() == 4:
        target_points = target_points[0]  # Take first batch
    
    pred_np = pred_points.cpu().numpy()
    target_np = target_points.cpu().numpy()
    
    seq_len = min(pred_np.shape[0], max_timesteps)
    
    # Create subplots
    fig = make_subplots(
        rows=seq_len, cols=2,
        subplot_titles=[f'Predicted t={t}' for t in range(seq_len)] + 
                      [f'Target t={t}' for t in range(seq_len)],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}] for _ in range(seq_len)]
    )
    
    for t in range(seq_len):
        # Predicted points
        fig.add_trace(
            go.Scatter3d(
                x=pred_np[t, :, 0],
                y=pred_np[t, :, 1],
                z=pred_np[t, :, 2],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.6),
                name=f'Pred t={t}'
            ),
            row=t+1, col=1
        )
        
        # Target points
        fig.add_trace(
            go.Scatter3d(
                x=target_np[t, :, 0],
                y=target_np[t, :, 1],
                z=target_np[t, :, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.6),
                name=f'Target t={t}'
            ),
            row=t+1, col=2
        )
    
    fig.update_layout(
        height=300 * seq_len,
        title_text="Point Cloud Sequence Visualization",
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()


def plot_training_curves(log_file: str, save_path: str = None) -> None:
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to log file
        save_path: Path to save plot
    """
    # This would typically read from tensorboard logs or a custom log format
    # For now, we'll create a placeholder function
    print(f"Plotting training curves from {log_file}")
    # Implementation would depend on the log format


def visualize_uncertainty(pred_points: torch.Tensor,
                         uncertainties: torch.Tensor,
                         save_path: str = None,
                         show: bool = True) -> None:
    """
    Visualize uncertainty estimates for point clouds.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3]
        uncertainties: Uncertainty estimates [seq_len, num_points]
        save_path: Path to save visualization
        show: Whether to show the plot
    """
    if pred_points.dim() == 4:
        pred_points = pred_points[0]  # Take first batch
    if uncertainties.dim() == 3:
        uncertainties = uncertainties[0]  # Take first batch
    
    pred_np = pred_points.cpu().numpy()
    unc_np = uncertainties.cpu().numpy()
    
    seq_len = pred_np.shape[0]
    
    # Create subplots
    fig = plt.figure(figsize=(15, 3 * seq_len))
    
    for t in range(seq_len):
        ax = fig.add_subplot(seq_len, 1, t+1, projection='3d')
        
        # Color points by uncertainty
        scatter = ax.scatter(pred_np[t, :, 0], pred_np[t, :, 1], pred_np[t, :, 2], 
                           c=unc_np[t], s=2, alpha=0.6, cmap='viridis')
        
        ax.set_title(f'Uncertainty Visualization t={t}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Uncertainty')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_animation(pred_points: torch.Tensor,
                   target_points: torch.Tensor,
                   save_path: str,
                   fps: int = 2) -> None:
    """
    Create animation of point cloud sequence.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3]
        target_points: Target point clouds [seq_len, num_points, 3]
        save_path: Path to save animation
        fps: Frames per second
    """
    if pred_points.dim() == 4:
        pred_points = pred_points[0]  # Take first batch
    if target_points.dim() == 4:
        target_points = target_points[0]  # Take first batch
    
    pred_np = pred_points.cpu().numpy()
    target_np = target_points.cpu().numpy()
    
    seq_len = pred_np.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    def animate(frame):
        plt.clf()
        
        # Predicted points
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pred_np[frame, :, 0], pred_np[frame, :, 1], pred_np[frame, :, 2], 
                   c='red', s=1, alpha=0.6)
        ax1.set_title(f'Predicted t={frame}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Target points
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(target_np[frame, :, 0], target_np[frame, :, 1], target_np[frame, :, 2], 
                   c='blue', s=1, alpha=0.6)
        ax2.set_title(f'Target t={frame}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=seq_len, interval=1000//fps, repeat=True)
    
    # Save animation
    anim.save(save_path, writer='pillow', fps=fps)
    
    plt.close()


def plot_metrics(metrics: Dict[str, List[float]], 
                save_path: str = None,
                show: bool = True) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary containing metric names and values
        save_path: Path to save plot
        show: Whether to show the plot
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
