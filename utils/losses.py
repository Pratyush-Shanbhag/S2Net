import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ChamferDistanceLoss(nn.Module):
    """
    Chamfer Distance loss for point cloud reconstruction.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(ChamferDistanceLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """
        Compute Chamfer Distance between predicted and target point clouds.
        
        Args:
            pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
            target_points: Target point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            Chamfer distance loss
        """
        batch_size, seq_len, num_points, _ = pred_points.shape
        
        total_loss = 0.0
        
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
            
            total_loss += chamfer_dist
        
        if self.reduction == 'mean':
            return total_loss.mean() / seq_len
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss / seq_len


class EarthMoverDistanceLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) loss for point cloud reconstruction.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(EarthMoverDistanceLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """
        Compute EMD between predicted and target point clouds.
        
        Args:
            pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
            target_points: Target point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            EMD loss
        """
        batch_size, seq_len, num_points, _ = pred_points.shape
        
        total_loss = 0.0
        
        for t in range(seq_len):
            pred_t = pred_points[:, t]  # [batch_size, num_points, 3]
            target_t = target_points[:, t]  # [batch_size, num_points, 3]
            
            # Compute pairwise distances
            pred_expanded = pred_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
            target_expanded = target_t.unsqueeze(1)  # [batch_size, 1, num_points, 3]
            
            distances = torch.norm(pred_expanded - target_expanded, dim=-1)  # [batch_size, num_points, num_points]
            
            # EMD is the minimum cost of matching points
            # Use Hungarian algorithm approximation with simple assignment
            emd_dist = self._approximate_emd(distances)
            
            total_loss += emd_dist
        
        if self.reduction == 'mean':
            return total_loss.mean() / seq_len
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss / seq_len
    
    def _approximate_emd(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Approximate EMD using greedy assignment.
        
        Args:
            distances: Pairwise distances [batch_size, num_points, num_points]
            
        Returns:
            Approximate EMD
        """
        batch_size, num_points, _ = distances.shape
        
        # Greedy assignment: assign each predicted point to its closest target point
        min_distances, _ = torch.min(distances, dim=2)  # [batch_size, num_points]
        
        return torch.mean(min_distances, dim=1)  # [batch_size]


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for variational autoencoders.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, 
                prior_mean: torch.Tensor, 
                prior_log_var: torch.Tensor,
                posterior_mean: torch.Tensor, 
                posterior_log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior distributions.
        
        Args:
            prior_mean: Prior means [batch_size, seq_len, latent_dim]
            prior_log_var: Prior log variances [batch_size, seq_len, latent_dim]
            posterior_mean: Posterior means [batch_size, seq_len, latent_dim]
            posterior_log_var: Posterior log variances [batch_size, seq_len, latent_dim]
            
        Returns:
            KL divergence loss
        """
        # KL divergence formula: 0.5 * sum(prior_log_var - posterior_log_var + 
        # (posterior_log_var.exp() + (posterior_mean - prior_mean).pow(2)) / prior_log_var.exp() - 1)
        
        kl_div = 0.5 * torch.sum(
            prior_log_var - posterior_log_var + 
            (posterior_log_var.exp() + (posterior_mean - prior_mean).pow(2)) / prior_log_var.exp() - 1,
            dim=-1
        )  # [batch_size, seq_len]
        
        if self.reduction == 'mean':
            return kl_div.mean()
        elif self.reduction == 'sum':
            return kl_div.sum()
        else:
            return kl_div


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss to encourage smooth predictions over time.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(TemporalConsistencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_points: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            Temporal consistency loss
        """
        if pred_points.shape[1] < 2:
            return torch.tensor(0.0, device=pred_points.device)
        
        # Compute differences between consecutive timesteps
        temporal_diff = pred_points[:, 1:] - pred_points[:, :-1]  # [batch_size, seq_len-1, num_points, 3]
        
        # L2 norm of temporal differences
        temporal_loss = torch.norm(temporal_diff, dim=-1)  # [batch_size, seq_len-1, num_points]
        
        if self.reduction == 'mean':
            return temporal_loss.mean()
        elif self.reduction == 'sum':
            return temporal_loss.sum()
        else:
            return temporal_loss


class UncertaintyLoss(nn.Module):
    """
    Loss for uncertainty estimation.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(UncertaintyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, 
                pred_points: torch.Tensor, 
                target_points: torch.Tensor,
                uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss.
        
        Args:
            pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
            target_points: Target point clouds [batch_size, seq_len, num_points, 3]
            uncertainties: Uncertainty estimates [batch_size, seq_len, num_points]
            
        Returns:
            Uncertainty-weighted loss
        """
        # Compute point-wise L2 loss
        point_loss = torch.norm(pred_points - target_points, dim=-1)  # [batch_size, seq_len, num_points]
        
        # Weight by uncertainty (higher uncertainty = lower weight)
        uncertainty_weights = 1.0 / (uncertainties + 1e-8)
        weighted_loss = point_loss * uncertainty_weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for S2Net training.
    """
    
    def __init__(self, 
                 kl_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 temporal_weight: float = 0.1,
                 uncertainty_weight: float = 0.1,
                 use_emd: bool = False,
                 use_temporal: bool = True,
                 use_uncertainty: bool = False):
        super(CombinedLoss, self).__init__()
        
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.temporal_weight = temporal_weight
        self.uncertainty_weight = uncertainty_weight
        
        # Initialize loss functions
        if use_emd:
            self.reconstruction_loss = EarthMoverDistanceLoss()
        else:
            self.reconstruction_loss = ChamferDistanceLoss()
        
        self.kl_loss = KLDivergenceLoss()
        
        if use_temporal:
            self.temporal_loss = TemporalConsistencyLoss()
        else:
            self.temporal_loss = None
        
        if use_uncertainty:
            self.uncertainty_loss = UncertaintyLoss()
        else:
            self.uncertainty_loss = None
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                target_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions dictionary
            target_points: Target point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            Dictionary containing individual loss components
        """
        pred_points = predictions['predicted_point_clouds']
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(pred_points, target_points)
        
        # KL divergence loss
        kl_loss = self.kl_loss(
            predictions['prior_mean'],
            predictions['prior_log_var'],
            predictions['posterior_mean'],
            predictions['posterior_log_var']
        )
        
        # Temporal consistency loss
        if self.temporal_loss is not None:
            temporal_loss = self.temporal_loss(pred_points)
        else:
            temporal_loss = torch.tensor(0.0, device=pred_points.device)
        
        # Uncertainty loss
        if self.uncertainty_loss is not None and 'uncertainties' in predictions:
            uncertainty_loss = self.uncertainty_loss(
                pred_points, target_points, predictions['uncertainties']
            )
        else:
            uncertainty_loss = torch.tensor(0.0, device=pred_points.device)
        
        # Combined loss
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.kl_weight * kl_loss +
                     self.temporal_weight * temporal_loss +
                     self.uncertainty_weight * uncertainty_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss,
            'uncertainty_loss': uncertainty_loss
        }
