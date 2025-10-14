import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointCloudDecoder(nn.Module):
    """
    Decoder for reconstructing point clouds from latent representations.
    Converts latent features back to 3D point coordinates.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 3,
                 num_points: int = 1024):
        super(PointCloudDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_points = num_points
        
        # Latent to feature mapping
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Point generation network
        self.point_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * output_dim)
        )
        
        # Optional refinement network
        self.refinement = nn.Sequential(
            nn.Linear(output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, 
                latent_features: torch.Tensor,
                use_refinement: bool = True) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            latent_features: Latent features [batch_size, seq_len, latent_dim]
            use_refinement: Whether to use refinement network
            
        Returns:
            point_clouds: Reconstructed point clouds [batch_size, seq_len, num_points, 3]
        """
        batch_size, seq_len, _ = latent_features.shape
        
        # Map latent features to hidden representation
        features = self.latent_to_features(latent_features)  # [batch_size, seq_len, hidden_dim]
        
        # Generate initial point coordinates
        point_coords = self.point_generator(features)  # [batch_size, seq_len, num_points * 3]
        point_coords = point_coords.view(batch_size, seq_len, self.num_points, self.output_dim)
        
        if use_refinement:
            # Refine point coordinates using refinement network
            refined_points = []
            for t in range(seq_len):
                # Get features and points for this timestep
                feat_t = features[:, t]  # [batch_size, hidden_dim]
                points_t = point_coords[:, t]  # [batch_size, num_points, 3]
                
                # Expand features to match number of points
                feat_expanded = feat_t.unsqueeze(1).expand(-1, self.num_points, -1)
                
                # Concatenate features with point coordinates
                refined_input = torch.cat([points_t, feat_expanded], dim=-1)
                
                # Apply refinement
                refined_t = self.refinement(refined_input)
                refined_points.append(refined_t)
            
            point_coords = torch.stack(refined_points, dim=1)
        
        return point_coords


class TemporalDecoder(nn.Module):
    """
    Temporal decoder that processes sequential latent features with LSTM.
    Better handles temporal dependencies in point cloud sequences.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 3,
                 num_points: int = 1024,
                 num_layers: int = 2):
        super(TemporalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_points = num_points
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Point generation network
        self.point_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * output_dim)
        )
        
        # Temporal consistency network
        self.temporal_consistency = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                latent_features: torch.Tensor,
                prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with temporal processing.
        
        Args:
            latent_features: Latent features [batch_size, seq_len, latent_dim]
            prev_hidden: Previous LSTM hidden state
            
        Returns:
            point_clouds: Reconstructed point clouds [batch_size, seq_len, num_points, 3]
            hidden_state: LSTM hidden state for next timestep
        """
        # Process through LSTM
        lstm_out, hidden_state = self.lstm(latent_features, prev_hidden)
        
        # Generate point clouds
        batch_size, seq_len, _ = lstm_out.shape
        point_coords = self.point_generator(lstm_out)
        point_coords = point_coords.view(batch_size, seq_len, self.num_points, self.output_dim)
        
        # Apply temporal consistency
        if seq_len > 1:
            # Concatenate current and previous features for consistency
            current_features = lstm_out[:, 1:, :]
            prev_features = lstm_out[:, :-1, :]
            
            # Pad to match sequence length
            prev_features = F.pad(prev_features, (0, 0, 1, 0), value=0)
            
            # Apply temporal consistency
            consistency_input = torch.cat([current_features, prev_features], dim=-1)
            consistency_features = self.temporal_consistency(consistency_input)
            
            # Update point generation with consistency features
            consistency_coords = self.point_generator(consistency_features)
            consistency_coords = consistency_coords.view(batch_size, seq_len, self.num_points, self.output_dim)
            
            # Blend original and consistency-based points
            point_coords = 0.7 * point_coords + 0.3 * consistency_coords
        
        return point_coords, hidden_state


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that generates point clouds at different resolutions.
    Helps with generating both fine and coarse details.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 3,
                 scales: list = [1024, 512, 256]):
        super(MultiScaleDecoder, self).__init__()
        
        self.scales = scales
        self.decoders = nn.ModuleList()
        
        for scale in scales:
            decoder = PointCloudDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_points=scale
            )
            self.decoders.append(decoder)
        
        # Fusion network to combine multi-scale outputs
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, scales[0] * output_dim)
        )
    
    def forward(self, 
                latent_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale generation.
        
        Args:
            latent_features: Latent features [batch_size, seq_len, latent_dim]
            
        Returns:
            point_clouds: Multi-scale point clouds [batch_size, seq_len, max_points, 3]
        """
        batch_size, seq_len, _ = latent_features.shape
        
        # Generate point clouds at different scales
        scale_outputs = []
        for decoder in self.decoders:
            scale_points = decoder(latent_features)
            scale_outputs.append(scale_points)
        
        # Upsample all scales to the finest scale
        max_points = self.scales[0]
        upsampled_outputs = []
        
        for i, scale_points in enumerate(scale_outputs):
            if scale_points.shape[2] < max_points:
                # Upsample to max_points
                upsampled = F.interpolate(
                    scale_points.view(batch_size * seq_len, -1, self.output_dim).transpose(1, 2),
                    size=max_points,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).view(batch_size, seq_len, max_points, self.output_dim)
            else:
                upsampled = scale_points
            upsampled_outputs.append(upsampled)
        
        # Concatenate all scales
        concatenated = torch.cat(upsampled_outputs, dim=-1)  # [batch_size, seq_len, max_points, 3 * num_scales]
        
        # Fuse multi-scale features
        fused_features = self.fusion(concatenated)
        fused_points = fused_features.view(batch_size, seq_len, max_points, self.output_dim)
        
        return fused_points


class UncertaintyDecoder(nn.Module):
    """
    Decoder that also predicts uncertainty estimates for each point.
    Useful for stochastic point cloud generation.
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 3,
                 num_points: int = 1024):
        super(UncertaintyDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_points = num_points
        
        # Point coordinate generation
        self.point_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points)
        )
    
    def forward(self, 
                latent_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            latent_features: Latent features [batch_size, seq_len, latent_dim]
            
        Returns:
            point_clouds: Generated point clouds [batch_size, seq_len, num_points, 3]
            uncertainties: Uncertainty estimates [batch_size, seq_len, num_points]
        """
        batch_size, seq_len, _ = latent_features.shape
        
        # Generate point coordinates
        point_coords = self.point_generator(latent_features)
        point_coords = point_coords.view(batch_size, seq_len, self.num_points, self.output_dim)
        
        # Estimate uncertainties
        uncertainties = self.uncertainty_estimator(latent_features)
        uncertainties = torch.sigmoid(uncertainties)  # Normalize to [0, 1]
        
        return point_coords, uncertainties
