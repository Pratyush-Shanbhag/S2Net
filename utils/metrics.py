import torch
import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def chamfer_distance(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between predicted and target point clouds.
    
    Args:
        pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [batch_size, seq_len, num_points, 3]
        
    Returns:
        Chamfer distance
    """
    # Ensure tensors have the same shape
    if pred_points.shape != target_points.shape:
        min_seq_len = min(pred_points.shape[1], target_points.shape[1])
        min_points = min(pred_points.shape[2], target_points.shape[2])
        pred_points = pred_points[:, :min_seq_len, :min_points, :]
        target_points = target_points[:, :min_seq_len, :min_points, :]
    
    batch_size, seq_len, num_points, _ = pred_points.shape
    
    total_distance = 0.0
    
    for t in range(seq_len):
        pred_t = pred_points[:, t]  # [batch_size, num_points, 3]
        target_t = target_points[:, t]  # [batch_size, num_points, 3]
        
        # Compute pairwise distances
        pred_expanded = pred_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
        target_expanded = target_t.unsqueeze(1)  # [batch_size, 1, num_points, 3]
        
        distances = torch.norm(pred_expanded - target_expanded, dim=-1)  # [batch_size, num_points, num_points]
        
        # Chamfer distance: min distance from each pred point to target + min distance from each target point to pred
        min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # [batch_size, num_points]
        min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # [batch_size, num_points]
        
        chamfer_dist = (torch.mean(min_dist_pred_to_target, dim=1) + 
                       torch.mean(min_dist_target_to_pred, dim=1))
        
        total_distance += chamfer_dist
    
    return total_distance / seq_len


def earth_mover_distance(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    Compute Earth Mover's Distance (EMD) between predicted and target point clouds.
    
    Args:
        pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [batch_size, seq_len, num_points, 3]
        
    Returns:
        EMD
    """
    batch_size, seq_len, num_points, _ = pred_points.shape
    
    total_distance = 0.0
    
    for t in range(seq_len):
        pred_t = pred_points[:, t].cpu().numpy()  # [batch_size, num_points, 3]
        target_t = target_points[:, t].cpu().numpy()  # [batch_size, num_points, 3]
        
        batch_emd = 0.0
        
        for b in range(batch_size):
            # Compute pairwise distances
            distances = cdist(pred_t[b], target_t[b])
            
            # Solve assignment problem using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(distances)
            
            # Compute EMD
            emd = distances[row_indices, col_indices].sum() / num_points
            batch_emd += emd
        
        total_distance += batch_emd / batch_size
    
    return torch.tensor(total_distance / seq_len, device=pred_points.device)


def hausdorff_distance(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    Compute Hausdorff Distance between predicted and target point clouds.
    
    Args:
        pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [batch_size, seq_len, num_points, 3]
        
    Returns:
        Hausdorff distance
    """
    # Ensure tensors have the same shape
    if pred_points.shape != target_points.shape:
        min_seq_len = min(pred_points.shape[1], target_points.shape[1])
        min_points = min(pred_points.shape[2], target_points.shape[2])
        pred_points = pred_points[:, :min_seq_len, :min_points, :]
        target_points = target_points[:, :min_seq_len, :min_points, :]
    
    batch_size, seq_len, num_points, _ = pred_points.shape
    
    total_distance = 0.0
    
    for t in range(seq_len):
        pred_t = pred_points[:, t]  # [batch_size, num_points, 3]
        target_t = target_points[:, t]  # [batch_size, num_points, 3]
        
        # Compute pairwise distances
        pred_expanded = pred_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
        target_expanded = target_t.unsqueeze(1)  # [batch_size, 1, num_points, 3]
        
        distances = torch.norm(pred_expanded - target_expanded, dim=-1)  # [batch_size, num_points, num_points]
        
        # Hausdorff distance: max of min distances
        min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # [batch_size, num_points]
        min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # [batch_size, num_points]
        
        hausdorff_dist = torch.max(
            torch.max(min_dist_pred_to_target, dim=1)[0],
            torch.max(min_dist_target_to_pred, dim=1)[0]
        )
        
        total_distance += hausdorff_dist
    
    return total_distance / seq_len


def point_cloud_density(points: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
    """
    Compute point cloud density.
    
    Args:
        points: Point clouds [batch_size, seq_len, num_points, 3]
        radius: Radius for density computation
        
    Returns:
        Density values
    """
    batch_size, seq_len, num_points, _ = points.shape
    
    total_density = 0.0
    
    for t in range(seq_len):
        points_t = points[:, t]  # [batch_size, num_points, 3]
        
        # Compute pairwise distances
        points_expanded = points_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
        distances = torch.norm(points_expanded - points_t.unsqueeze(1), dim=-1)  # [batch_size, num_points, num_points]
        
        # Count points within radius (excluding self)
        mask = (distances < radius) & (distances > 0)
        density = torch.sum(mask.float(), dim=2)  # [batch_size, num_points]
        
        total_density += torch.mean(density, dim=1)  # [batch_size]
    
    return total_density / seq_len


def temporal_consistency(pred_points: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal consistency of predicted point clouds.
    
    Args:
        pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
        
    Returns:
        Temporal consistency score
    """
    if pred_points.shape[1] < 2:
        return torch.tensor(0.0, device=pred_points.device)
    
    # Compute differences between consecutive timesteps
    temporal_diff = pred_points[:, 1:] - pred_points[:, :-1]  # [batch_size, seq_len-1, num_points, 3]
    
    # L2 norm of temporal differences
    temporal_norm = torch.norm(temporal_diff, dim=-1)  # [batch_size, seq_len-1, num_points]
    
    # Temporal consistency is inverse of temporal variation
    consistency = 1.0 / (1.0 + torch.mean(temporal_norm, dim=(1, 2)))
    
    return consistency


def compute_metrics(pred_points: torch.Tensor, target_points: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive metrics for point cloud prediction.
    
    Args:
        pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
        target_points: Target point clouds [batch_size, seq_len, num_points, 3]
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Chamfer Distance
    metrics['chamfer_distance'] = chamfer_distance(pred_points, target_points).mean().item()
    
    # Hausdorff Distance
    metrics['hausdorff_distance'] = hausdorff_distance(pred_points, target_points).mean().item()
    
    # Temporal consistency
    metrics['temporal_consistency'] = temporal_consistency(pred_points).mean().item()
    
    # Point cloud density
    pred_density = point_cloud_density(pred_points).mean().item()
    target_density = point_cloud_density(target_points).mean().item()
    metrics['pred_density'] = pred_density
    metrics['target_density'] = target_density
    metrics['density_ratio'] = pred_density / (target_density + 1e-8)
    
    # Ensure tensors have the same shape - handle both sequence length and point count mismatches
    if pred_points.shape != target_points.shape:
        # Resize to match the smaller dimensions
        min_seq_len = min(pred_points.shape[1], target_points.shape[1])
        min_points = min(pred_points.shape[2], target_points.shape[2])
        pred_points = pred_points[:, :min_seq_len, :min_points, :]
        target_points = target_points[:, :min_seq_len, :min_points, :]
    
    l2_distance = torch.norm(pred_points - target_points, dim=-1).mean().item()
    metrics['l2_distance'] = l2_distance
    
    # Mean absolute error
    mae = torch.mean(torch.abs(pred_points - target_points)).item()
    metrics['mae'] = mae
    
    # Root mean square error
    rmse = torch.sqrt(torch.mean((pred_points - target_points) ** 2)).item()
    metrics['rmse'] = rmse
    
    return metrics


def compute_sequence_metrics(pred_sequences: torch.Tensor, 
                           target_sequences: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for entire sequences.
    
    Args:
        pred_sequences: Predicted sequences [batch_size, num_samples, seq_len, num_points, 3]
        target_sequences: Target sequences [batch_size, seq_len, num_points, 3]
        
    Returns:
        Dictionary containing sequence-level metrics
    """
    batch_size, num_samples, seq_len, num_points, _ = pred_sequences.shape
    
    # Compute metrics for each sample
    sample_metrics = []
    for s in range(num_samples):
        sample_pred = pred_sequences[:, s]  # [batch_size, seq_len, num_points, 3]
        sample_metrics.append(compute_metrics(sample_pred, target_sequences))
    
    # Average across samples
    avg_metrics = {}
    for key in sample_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in sample_metrics])
    
    # Compute sample diversity (if multiple samples)
    if num_samples > 1:
        diversity_scores = []
        for b in range(batch_size):
            batch_samples = pred_sequences[b]  # [num_samples, seq_len, num_points, 3]
            
            # Compute pairwise distances between samples
            sample_distances = []
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    dist = chamfer_distance(
                        batch_samples[i:i+1], 
                        batch_samples[j:j+1]
                    ).item()
                    sample_distances.append(dist)
            
            diversity_scores.append(np.mean(sample_distances))
        
        avg_metrics['sample_diversity'] = np.mean(diversity_scores)
    
    return avg_metrics


def compute_chamfer_distance(pred_points: np.ndarray, target_points: np.ndarray) -> float:
    """
    Compute Chamfer Distance between predicted and target point clouds using numpy.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3]
        target_points: Target point clouds [seq_len, num_points, 3]
        
    Returns:
        Chamfer distance
    """
    total_distance = 0.0
    seq_len = min(pred_points.shape[0], target_points.shape[0])
    
    for t in range(seq_len):
        pred_t = pred_points[t]  # [num_points, 3]
        target_t = target_points[t]  # [num_points, 3]
        
        # Compute pairwise distances
        pred_expanded = pred_t[:, np.newaxis, :]  # [num_points, 1, 3]
        target_expanded = target_t[np.newaxis, :, :]  # [1, num_points, 3]
        
        distances = np.linalg.norm(pred_expanded - target_expanded, axis=-1)  # [num_points, num_points]
        
        # Chamfer distance: min distance from each pred point to target + min distance from each target point to pred
        min_dist_pred_to_target = np.min(distances, axis=1)  # [num_points]
        min_dist_target_to_pred = np.min(distances, axis=0)  # [num_points]
        
        chamfer_dist = (np.mean(min_dist_pred_to_target) + np.mean(min_dist_target_to_pred))
        total_distance += chamfer_dist
    
    return total_distance / seq_len


def compute_hausdorff_distance(pred_points: np.ndarray, target_points: np.ndarray) -> float:
    """
    Compute Hausdorff Distance between predicted and target point clouds using numpy.
    
    Args:
        pred_points: Predicted point clouds [seq_len, num_points, 3]
        target_points: Target point clouds [seq_len, num_points, 3]
        
    Returns:
        Hausdorff distance
    """
    total_distance = 0.0
    seq_len = min(pred_points.shape[0], target_points.shape[0])
    
    for t in range(seq_len):
        pred_t = pred_points[t]  # [num_points, 3]
        target_t = target_points[t]  # [num_points, 3]
        
        # Compute pairwise distances
        pred_expanded = pred_t[:, np.newaxis, :]  # [num_points, 1, 3]
        target_expanded = target_t[np.newaxis, :, :]  # [1, num_points, 3]
        
        distances = np.linalg.norm(pred_expanded - target_expanded, axis=-1)  # [num_points, num_points]
        
        # Hausdorff distance: max of min distances
        min_dist_pred_to_target = np.min(distances, axis=1)  # [num_points]
        min_dist_target_to_pred = np.min(distances, axis=0)  # [num_points]
        
        hausdorff_dist = max(np.max(min_dist_pred_to_target), np.max(min_dist_target_to_pred))
        total_distance += hausdorff_dist
    
    return total_distance / seq_len


def evaluate_model(model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  device: str,
                  num_samples: int = 1) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to use
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            
            # Generate predictions
            if num_samples == 1:
                predictions = model(input_sequence, is_training=False)
                pred_points = predictions['predicted_point_clouds']
                metrics = compute_metrics(pred_points, target_sequence)
            else:
                # Generate multiple samples
                pred_sequences = model.sample_future(
                    input_sequence, 
                    future_steps=target_sequence.shape[1],
                    num_samples=num_samples
                )
                metrics = compute_sequence_metrics(pred_sequences, target_sequence)
            
            all_metrics.append(metrics)
    
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics
