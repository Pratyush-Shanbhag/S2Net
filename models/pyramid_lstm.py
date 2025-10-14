import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PyramidLSTM(nn.Module):
    """
    Pyramid-LSTM architecture with skip connections for enhanced temporal modeling.
    Processes features at multiple scales with LSTM layers and skip connections.
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 256,
                 num_levels: int = 3,
                 dropout: float = 0.1):
        super(PyramidLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Create LSTM layers for each pyramid level
        self.lstm_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for level in range(num_levels):
            # Each level has different hidden dimensions
            level_hidden_dim = hidden_dim // (2 ** level)
            
            # LSTM layer for this level
            lstm = nn.LSTM(
                input_size=input_dim if level == 0 else hidden_dim // (2 ** (level - 1)),
                hidden_size=level_hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            self.lstm_layers.append(lstm)
            
            # Skip connection projection
            if level > 0:
                skip_proj = nn.Linear(
                    hidden_dim // (2 ** (level - 1)) + level_hidden_dim,
                    level_hidden_dim
                )
                self.skip_connections.append(skip_proj)
            else:
                self.skip_connections.append(None)
        
        # Fusion layer to combine multi-level features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                features: torch.Tensor,
                encoder_hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the pyramid-LSTM.
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            encoder_hidden_states: Optional hidden states from encoder for skip connections
            
        Returns:
            output_features: Processed features [batch_size, seq_len, hidden_dim]
            hidden_states: Hidden states from all LSTM levels
        """
        batch_size, seq_len, _ = features.shape
        
        # Process through each pyramid level
        level_outputs = []
        all_hidden_states = []
        current_features = features
        
        for level in range(self.num_levels):
            # Process through LSTM at this level
            lstm_out, hidden_states = self.lstm_layers[level](current_features)
            all_hidden_states.append(hidden_states)
            
            # Apply skip connection if available
            if level > 0 and encoder_hidden_states is not None:
                # Get corresponding encoder features for skip connection
                encoder_features = self._get_encoder_features_at_level(encoder_hidden_states, level)
                
                # Concatenate with current LSTM output
                skip_input = torch.cat([lstm_out, encoder_features], dim=-1)
                
                # Apply skip connection projection
                lstm_out = self.skip_connections[level](skip_input)
            
            level_outputs.append(lstm_out)
            
            # Downsample features for next level (except last level)
            if level < self.num_levels - 1:
                # Simple downsampling by taking every 2nd timestep
                current_features = lstm_out[:, ::2, :]
        
        # Upsample and combine features from all levels
        combined_features = self._combine_pyramid_features(level_outputs, seq_len)
        
        # Apply fusion layer
        output_features = self.fusion(combined_features)
        
        return output_features, all_hidden_states
    
    def _get_encoder_features_at_level(self, 
                                     encoder_hidden_states: List[Tuple[torch.Tensor, torch.Tensor]], 
                                     level: int) -> torch.Tensor:
        """
        Extract features from encoder hidden states at a specific pyramid level.
        """
        if level < len(encoder_hidden_states):
            # Use the hidden state from the corresponding encoder level
            h_n, c_n = encoder_hidden_states[level]
            # Take the last layer's hidden state
            return h_n[-1].unsqueeze(1).expand(-1, encoder_hidden_states[0][0].shape[1], -1)
        else:
            # If no corresponding level, use the last available level
            h_n, c_n = encoder_hidden_states[-1]
            return h_n[-1].unsqueeze(1).expand(-1, encoder_hidden_states[0][0].shape[1], -1)
    
    def _combine_pyramid_features(self, 
                                 level_outputs: List[torch.Tensor], 
                                 target_seq_len: int) -> torch.Tensor:
        """
        Combine features from all pyramid levels.
        """
        # Start with the finest level (level 0)
        combined = level_outputs[0]
        
        # Upsample and add features from coarser levels
        for level in range(1, len(level_outputs)):
            level_features = level_outputs[level]
            
            # Upsample to match target sequence length
            upsampled = F.interpolate(
                level_features.transpose(1, 2),  # [batch, hidden, seq]
                size=target_seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, seq, hidden]
            
            # Add to combined features
            combined = combined + upsampled
        
        return combined


class TemporalAlignmentModule(nn.Module):
    """
    Module for temporal alignment of features across different pyramid levels.
    Ensures that features at different scales are temporally aligned.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_levels: int = 3):
        super(TemporalAlignmentModule, self).__init__()
        
        self.num_levels = num_levels
        
        # Temporal alignment networks for each level
        self.alignment_nets = nn.ModuleList()
        
        for level in range(num_levels):
            alignment_net = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
            self.alignment_nets.append(alignment_net)
        
        # Global temporal alignment
        self.global_alignment = nn.Sequential(
            nn.Linear(feature_dim * num_levels, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, 
                pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Align features from different pyramid levels temporally.
        
        Args:
            pyramid_features: List of features from each pyramid level
            
        Returns:
            aligned_features: Temporally aligned features
        """
        aligned_features = []
        
        for level, features in enumerate(pyramid_features):
            # Apply level-specific alignment
            aligned = self.alignment_nets[level](features)
            aligned_features.append(aligned)
        
        # Concatenate all aligned features
        concatenated = torch.cat(aligned_features, dim=-1)
        
        # Apply global alignment
        final_features = self.global_alignment(concatenated)
        
        return final_features


class SkipConnectionModule(nn.Module):
    """
    Module for implementing skip connections between encoder and decoder.
    Helps preserve fine-grained details in the predictions.
    """
    
    def __init__(self, 
                 encoder_dim: int = 256,
                 decoder_dim: int = 256,
                 output_dim: int = 256):
        super(SkipConnectionModule, self).__init__()
        
        self.encoder_proj = nn.Linear(encoder_dim, output_dim)
        self.decoder_proj = nn.Linear(decoder_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, 
                encoder_features: torch.Tensor, 
                decoder_features: torch.Tensor) -> torch.Tensor:
        """
        Apply skip connection between encoder and decoder features.
        
        Args:
            encoder_features: Features from encoder
            decoder_features: Features from decoder
            
        Returns:
            fused_features: Fused features with skip connection
        """
        # Project both features to same dimension
        enc_proj = self.encoder_proj(encoder_features)
        dec_proj = self.decoder_proj(decoder_features)
        
        # Concatenate and fuse
        concatenated = torch.cat([enc_proj, dec_proj], dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
