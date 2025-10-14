import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class PointCloudEncoder(nn.Module):
    """
    LSTM-based encoder for processing sequential point cloud data.
    Converts point clouds to feature representations for the variational module.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(PointCloudEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Point cloud feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            point_clouds: Input point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            features: Encoded features [batch_size, seq_len, hidden_dim]
            hidden_states: LSTM hidden states (h_n, c_n)
        """
        batch_size, seq_len, num_points, _ = point_clouds.shape
        
        # Process each point cloud in the sequence
        features = []
        for t in range(seq_len):
            # Extract features from point cloud at time t
            pc_t = point_clouds[:, t]  # [batch_size, num_points, 3]
            
            # Apply MLP to each point
            pc_features = self.point_mlp(pc_t)  # [batch_size, num_points, hidden_dim]
            
            # Global pooling (mean) to get sequence-level features
            global_features = torch.mean(pc_features, dim=1)  # [batch_size, hidden_dim]
            features.append(global_features)
        
        # Stack features along sequence dimension
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Apply output projection
        output_features = self.output_proj(lstm_out)
        
        return output_features, (h_n, c_n)


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder that processes point clouds at different resolutions.
    Used in the pyramid-LSTM architecture.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dim: int = 256,
                 scales: List[int] = [1, 2, 4]):
        super(MultiScaleEncoder, self).__init__()
        
        self.scales = scales
        self.encoders = nn.ModuleList()
        
        for scale in scales:
            encoder = PointCloudEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim // len(scales),
                num_layers=2
            )
            self.encoders.append(encoder)
        
        # Fusion layer
        total_hidden_dim = hidden_dim * len(scales)
        self.fusion = nn.Linear(total_hidden_dim, hidden_dim)
        
    def forward(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with multi-scale processing.
        
        Args:
            point_clouds: Input point clouds [batch_size, seq_len, num_points, 3]
            
        Returns:
            fused_features: Multi-scale fused features
            hidden_states: List of hidden states from each scale
        """
        batch_size, seq_len, num_points, _ = point_clouds.shape
        scale_features = []
        all_hidden_states = []
        
        for i, (scale, encoder) in enumerate(zip(self.scales, self.encoders)):
            # Downsample point clouds
            if scale > 1:
                # Simple downsampling by taking every scale-th point
                downsampled_pc = point_clouds[:, :, ::scale, :]
            else:
                downsampled_pc = point_clouds
            
            # Encode at this scale
            features, hidden_states = encoder(downsampled_pc)
            scale_features.append(features)
            all_hidden_states.append(hidden_states)
        
        # Concatenate features from all scales
        concatenated_features = torch.cat(scale_features, dim=-1)  # [batch_size, seq_len, hidden_dim]
        
        # Fuse multi-scale features
        fused_features = self.fusion(concatenated_features)
        
        return fused_features, all_hidden_states
