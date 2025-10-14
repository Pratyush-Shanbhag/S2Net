import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .encoder import PointCloudEncoder, MultiScaleEncoder
from .variational import ConditionalVariationalModule, TemporalVariationalModule
from .pyramid_lstm import PyramidLSTM, TemporalAlignmentModule, SkipConnectionModule
from .decoder import PointCloudDecoder, TemporalDecoder, MultiScaleDecoder, UncertaintyDecoder


class S2Net(nn.Module):
    """
    S2Net: Stochastic Sequential Pointcloud Forecasting
    
    A conditional variational recurrent neural network with pyramid-LSTM structure
    for predicting future LiDAR point clouds in autonomous driving scenarios.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 num_points: int = 1024,
                 num_lstm_layers: int = 2,
                 num_pyramid_levels: int = 3,
                 use_temporal_variational: bool = True,
                 use_multi_scale: bool = True,
                 use_uncertainty: bool = True,
                 use_temporal_decoder: bool = True,
                 dropout: float = 0.1):
        super(S2Net, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.use_temporal_variational = use_temporal_variational
        self.use_multi_scale = use_multi_scale
        self.use_uncertainty = use_uncertainty
        self.use_temporal_decoder = use_temporal_decoder
        
        # Encoder
        if use_multi_scale:
            self.encoder = MultiScaleEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                scales=[1, 2, 4]
            )
        else:
            self.encoder = PointCloudEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_lstm_layers,
                dropout=dropout
            )
        
        # Variational Module
        if use_temporal_variational:
            self.variational = TemporalVariationalModule(
                input_dim=hidden_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim
            )
        else:
            self.variational = ConditionalVariationalModule(
                input_dim=hidden_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim
            )
        
        # Pyramid-LSTM
        self.pyramid_lstm = PyramidLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_levels=num_pyramid_levels,
            dropout=dropout
        )
        
        # Temporal Alignment
        self.temporal_alignment = TemporalAlignmentModule(
            feature_dim=hidden_dim,
            num_levels=num_pyramid_levels
        )
        
        # Skip Connection Module
        self.skip_connection = SkipConnectionModule(
            encoder_dim=hidden_dim,
            decoder_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Decoder
        if use_uncertainty:
            self.decoder = UncertaintyDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                num_points=num_points
            )
        elif use_multi_scale:
            self.decoder = MultiScaleDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                scales=[num_points, num_points // 2, num_points // 4]
            )
        elif use_temporal_decoder:
            self.decoder = TemporalDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                num_points=num_points,
                num_layers=num_lstm_layers
            )
        else:
            self.decoder = PointCloudDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                num_points=num_points
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                input_point_clouds: torch.Tensor,
                is_training: bool = True,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of S2Net.
        
        Args:
            input_point_clouds: Input point cloud sequence [batch_size, seq_len, num_points, 3]
            is_training: Whether in training mode
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing:
                - predicted_point_clouds: Predicted point clouds
                - latent_samples: Sampled latent variables
                - prior_mean: Prior means
                - prior_log_var: Prior log variances
                - posterior_mean: Posterior means
                - posterior_log_var: Posterior log variances
                - uncertainties: Uncertainty estimates (if return_uncertainty=True)
        """
        batch_size, seq_len, num_points, _ = input_point_clouds.shape
        
        # Encode input point clouds
        if self.use_multi_scale:
            encoder_features, encoder_hidden_states = self.encoder(input_point_clouds)
        else:
            encoder_features, encoder_hidden_states = self.encoder(input_point_clouds)
            encoder_hidden_states = [encoder_hidden_states]  # Wrap in list for consistency
        
        # Process through variational module
        if self.use_temporal_variational:
            (latent_samples, prior_mean, prior_log_var, 
             posterior_mean, posterior_log_var) = self.variational(
                encoder_features, is_training=is_training
            )
        else:
            (latent_samples, prior_mean, prior_log_var, 
             posterior_mean, posterior_log_var) = self.variational(
                encoder_features, is_training=is_training
            )
        
        # Process through pyramid-LSTM
        pyramid_features, pyramid_hidden_states = self.pyramid_lstm(
            encoder_features, encoder_hidden_states
        )
        
        # Apply temporal alignment
        aligned_features = self.temporal_alignment([pyramid_features])
        
        # Apply skip connections
        skip_features = self.skip_connection(encoder_features, aligned_features)
        
        # Combine features
        combined_features = encoder_features + skip_features
        combined_features = self.output_proj(combined_features)
        
        # Decode to point clouds
        if self.use_uncertainty:
            predicted_point_clouds, uncertainties = self.decoder(latent_samples)
        else:
            predicted_point_clouds = self.decoder(latent_samples)
            uncertainties = None
        
        # Prepare output
        output = {
            'predicted_point_clouds': predicted_point_clouds,
            'latent_samples': latent_samples,
            'prior_mean': prior_mean,
            'prior_log_var': prior_log_var,
            'posterior_mean': posterior_mean,
            'posterior_log_var': posterior_log_var
        }
        
        if return_uncertainty and uncertainties is not None:
            output['uncertainties'] = uncertainties
        
        return output
    
    def sample_future(self, 
                     input_point_clouds: torch.Tensor,
                     future_steps: int = 10,
                     num_samples: int = 1) -> torch.Tensor:
        """
        Sample future point clouds from the model.
        
        Args:
            input_point_clouds: Input point cloud sequence [batch_size, seq_len, num_points, 3]
            future_steps: Number of future steps to predict
            num_samples: Number of samples to generate
            
        Returns:
            future_point_clouds: Sampled future point clouds [batch_size, num_samples, future_steps, num_points, 3]
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = input_point_clouds.shape[0]
            
            # Encode input sequence
            if self.use_multi_scale:
                encoder_features, encoder_hidden_states = self.encoder(input_point_clouds)
            else:
                encoder_features, encoder_hidden_states = self.encoder(input_point_clouds)
                encoder_hidden_states = [encoder_hidden_states]
            
            # Get latent representation of the last timestep
            last_features = encoder_features[:, -1:, :]  # [batch_size, 1, hidden_dim]
            
            # Generate future latent samples
            future_latents = []
            current_latent = None
            
            for step in range(future_steps):
                # Sample from variational module
                if self.use_temporal_variational:
                    (latent_samples, _, _, _, _) = self.variational(
                        last_features, is_training=False
                    )
                else:
                    (latent_samples, _, _, _, _) = self.variational(
                        last_features, prev_latent=current_latent, is_training=False
                    )
                
                future_latents.append(latent_samples)
                current_latent = latent_samples[:, -1, :]  # Use last timestep for next iteration
                
                # Update features for next iteration (simple approach)
                last_features = encoder_features[:, -1:, :]  # Keep using last input features
            
            # Stack future latents
            future_latents = torch.cat(future_latents, dim=1)  # [batch_size, future_steps, latent_dim]
            
            # Decode to point clouds
            if self.use_uncertainty:
                future_point_clouds, _ = self.decoder(future_latents)
            else:
                future_point_clouds = self.decoder(future_latents)
            
            # Repeat for multiple samples
            future_point_clouds = future_point_clouds.unsqueeze(1).repeat(1, num_samples, 1, 1, 1)
            
        return future_point_clouds
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    target_point_clouds: torch.Tensor,
                    kl_weight: float = 1.0,
                    reconstruction_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss for training.
        
        Args:
            predictions: Model predictions from forward pass
            target_point_clouds: Target point clouds [batch_size, seq_len, num_points, 3]
            kl_weight: Weight for KL divergence loss
            reconstruction_weight: Weight for reconstruction loss
            
        Returns:
            Dictionary containing individual loss components
        """
        # Reconstruction loss (Chamfer Distance)
        reconstruction_loss = self._chamfer_distance(
            predictions['predicted_point_clouds'],
            target_point_clouds
        )
        
        # KL divergence loss
        kl_loss = self.variational.compute_kl_divergence(
            predictions['prior_mean'],
            predictions['prior_log_var'],
            predictions['posterior_mean'],
            predictions['posterior_log_var']
        )
        
        # Total loss
        total_loss = (reconstruction_weight * reconstruction_loss + 
                     kl_weight * kl_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }
    
    def _chamfer_distance(self, 
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
        
        for t in range(min(seq_len, target_points.shape[1])):
            pred_t = pred_points[:, t]  # [batch_size, num_points, 3]
            target_t = target_points[:, t]  # [batch_size, num_points, 3]
            
            # Compute pairwise distances
            pred_expanded = pred_t.unsqueeze(2)  # [batch_size, num_points, 1, 3]
            target_expanded = target_t.unsqueeze(1)  # [batch_size, 1, num_points, 3]
            
            distances = torch.norm(pred_expanded - target_expanded, dim=-1)  # [batch_size, num_points, num_points]
            
            # Chamfer distance: min distance from each pred point to target + min distance from each target point to pred
            min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # [batch_size, num_points]
            min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # [batch_size, num_points]
            
            chamfer_dist = (torch.mean(min_dist_pred_to_target) + 
                           torch.mean(min_dist_target_to_pred))
            
            total_loss += chamfer_dist
        
        return total_loss / min(seq_len, target_points.shape[1])
    
    def get_model_size(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'num_points': self.num_points,
                'use_temporal_variational': self.use_temporal_variational,
                'use_multi_scale': self.use_multi_scale,
                'use_uncertainty': self.use_uncertainty
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cuda'):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
