import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ImprovedChamferDistanceLoss(nn.Module):
    """
    Improved Chamfer Distance implementation with better numerical stability
    and proper handling of point cloud distances as described in the S2Net paper.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(ImprovedChamferDistanceLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, 
                pred_points: torch.Tensor, 
                target_points: torch.Tensor) -> torch.Tensor:
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
        num_valid_frames = 0
        
        for t in range(min(seq_len, target_points.shape[1])):
            pred_t = pred_points[:, t]  # [batch_size, num_points, 3]
            target_t = target_points[:, t]  # [batch_size, num_points, 3]
            
            # Compute pairwise squared distances
            pred_expanded = pred_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
            target_expanded = target_t.unsqueeze(1)  # [batch_size, 1, num_points, 3]
            
            # Squared distances for numerical stability
            squared_distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)
            
            # Chamfer distance components
            min_dist_pred_to_target = torch.min(squared_distances, dim=2)[0]  # [batch_size, num_points]
            min_dist_target_to_pred = torch.min(squared_distances, dim=1)[0]  # [batch_size, num_points]
            
            # Convert to actual distances with numerical stability
            dist_pred_to_target = torch.sqrt(min_dist_pred_to_target + 1e-8)
            dist_target_to_pred = torch.sqrt(min_dist_target_to_pred + 1e-8)
            
            # Chamfer distance for this timestep
            chamfer_dist = torch.mean(dist_pred_to_target) + torch.mean(dist_target_to_pred)
            
            total_loss += chamfer_dist
            num_valid_frames += 1
        
        if num_valid_frames == 0:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        return total_loss / num_valid_frames


class TwoStreamLoss(nn.Module):
    """
    Combined loss function for two-stream S2Net architecture.
    Properly handles deterministic and stochastic stream losses.
    """
    
    def __init__(self, 
                 kl_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 deterministic_weight: float = 0.5,
                 stochastic_weight: float = 0.5,
                 use_annealing: bool = True,
                 annealing_steps: int = 1000):
        super(TwoStreamLoss, self).__init__()
        
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.deterministic_weight = deterministic_weight
        self.stochastic_weight = stochastic_weight
        self.use_annealing = use_annealing
        self.annealing_steps = annealing_steps
        
        # Loss functions
        self.chamfer_loss = ImprovedChamferDistanceLoss()
        self.kl_loss = KLDivergenceLoss()
        
        # Annealing parameters
        self.step_count = 0
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                target_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for two-stream architecture.
        
        Args:
            predictions: Model predictions dictionary
            target_points: Target point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            Dictionary containing individual loss components
        """
        # Deterministic stream loss
        deterministic_loss = self.chamfer_loss(
            predictions['deterministic_predictions'], 
            target_points
        )
        
        # Stochastic stream loss
        stochastic_loss = self.chamfer_loss(
            predictions['stochastic_predictions'], 
            target_points
        )
        
        # KL divergence loss (only for stochastic stream)
        kl_loss = self.kl_loss(
            predictions['prior_mean'],
            predictions['prior_log_var'],
            predictions['posterior_mean'],
            predictions['posterior_log_var']
        )
        
        # Apply KL annealing if enabled
        if self.use_annealing:
            annealing_factor = min(1.0, self.step_count / self.annealing_steps)
            kl_weight = self.kl_weight * annealing_factor
        else:
            kl_weight = self.kl_weight
        
        # Combined reconstruction loss
        reconstruction_loss = (self.deterministic_weight * deterministic_loss + 
                             self.stochastic_weight * stochastic_loss)
        
        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     kl_weight * kl_loss)
        
        # Update step count for annealing
        self.step_count += 1
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'deterministic_loss': deterministic_loss,
            'stochastic_loss': stochastic_loss,
            'kl_loss': kl_loss,
            'kl_weight': kl_weight
        }


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
        # KL divergence: KL(q(z|x) || p(z))
        kl_div = 0.5 * torch.sum(
            prior_log_var - posterior_log_var + 
            (posterior_log_var.exp() + (posterior_mean - prior_mean).pow(2)) / 
            (prior_log_var.exp() + 1e-8) - 1,
            dim=-1
        )
        
        if self.reduction == 'mean':
            return kl_div.mean()
        elif self.reduction == 'sum':
            return kl_div.sum()
        else:
            return kl_div


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss to ensure smooth transitions between frames.
    """
    
    def __init__(self, weight: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred_points: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            pred_points: Predicted point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            Temporal consistency loss
        """
        if pred_points.shape[1] < 2:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        # Compute differences between consecutive frames
        frame_diffs = pred_points[:, 1:] - pred_points[:, :-1]
        
        # L2 norm of differences
        temporal_loss = torch.mean(torch.norm(frame_diffs, dim=-1))
        
        return self.weight * temporal_loss


class CombinedTwoStreamLoss(nn.Module):
    """
    Complete loss function combining all components for two-stream S2Net.
    """
    
    def __init__(self, 
                 kl_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 deterministic_weight: float = 0.5,
                 stochastic_weight: float = 0.5,
                 temporal_weight: float = 0.1,
                 use_annealing: bool = True,
                 annealing_steps: int = 1000):
        super(CombinedTwoStreamLoss, self).__init__()
        
        self.two_stream_loss = TwoStreamLoss(
            kl_weight=kl_weight,
            reconstruction_weight=reconstruction_weight,
            deterministic_weight=deterministic_weight,
            stochastic_weight=stochastic_weight,
            use_annealing=use_annealing,
            annealing_steps=annealing_steps
        )
        
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                target_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute complete loss for two-stream S2Net.
        """
        # Two-stream losses
        two_stream_losses = self.two_stream_loss(predictions, target_points)
        
        # Temporal consistency loss (on stochastic predictions)
        temporal_loss = self.temporal_loss(predictions['stochastic_predictions'])
        
        # Total loss
        total_loss = two_stream_losses['total_loss'] + temporal_loss
        
        # Combine all losses
        result = {
            'total_loss': total_loss,
            'temporal_loss': temporal_loss
        }
        result.update(two_stream_losses)
        
        return result
