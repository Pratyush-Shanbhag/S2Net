#!/usr/bin/env python3
"""
KITTI dataset loader for TwoStreamS2Net training.
Handles sequences 1-5 for training and 6-7 for validation.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import random


class KITTIDataset(Dataset):
    """KITTI dataset for point cloud sequence prediction."""
    
    def __init__(self, 
                 data_path: str,
                 sequences: List[str],
                 sequence_length: int = 5,
                 prediction_length: int = 3,
                 num_points: int = 512,
                 max_sequence_length: int = 500,
                 is_training: bool = True):
        """
        Initialize KITTI dataset.
        
        Args:
            data_path: Path to KITTI sequences directory
            sequences: List of sequence IDs to use (e.g., ['01', '02', '03'])
            sequence_length: Input sequence length
            prediction_length: Future prediction length
            num_points: Number of points per point cloud
            max_sequence_length: Maximum length of sequences to process
            is_training: Whether this is for training (affects augmentation)
        """
        self.data_path = data_path
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_points = num_points
        self.max_sequence_length = max_sequence_length
        self.is_training = is_training
        
        # Load all point cloud files from specified sequences
        self.point_cloud_files = []
        self.sequence_info = []
        
        for seq_id in sequences:
            seq_path = os.path.join(data_path, seq_id, 'velodyne')
            if os.path.exists(seq_path):
                files = sorted(glob.glob(os.path.join(seq_path, '*.bin')))
                if len(files) > 0:
                    # Limit sequence length for memory efficiency
                    files = files[:min(len(files), max_sequence_length)]
                    self.point_cloud_files.extend(files)
                    self.sequence_info.extend([seq_id] * len(files))
        
        print(f"Loaded {len(self.point_cloud_files)} point cloud files from sequences {sequences}")
        
        # Create valid sequence indices
        self.valid_indices = []
        for i in range(len(self.point_cloud_files) - sequence_length - prediction_length + 1):
            # Check if all files in the sequence are from the same sequence ID
            start_seq = self.sequence_info[i]
            end_seq = self.sequence_info[i + sequence_length + prediction_length - 1]
            if start_seq == end_seq:  # All files from same sequence
                self.valid_indices.append(i)
        
        print(f"Created {len(self.valid_indices)} valid sequences")
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from .bin file."""
        try:
            # Read binary file
            points = np.fromfile(file_path, dtype=np.float32)
            
            # Reshape to (N, 4) - x, y, z, intensity
            points = points.reshape(-1, 4)
            
            # Take only x, y, z coordinates
            points = points[:, :3]
            
            # Remove points that are too far (likely noise)
            distances = np.linalg.norm(points, axis=1)
            points = points[distances < 50.0]  # Keep points within 50m
            
            # Subsample to num_points if necessary
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                # Pad with zeros if not enough points
                padding = np.zeros((self.num_points - len(points), 3))
                points = np.vstack([points, padding])
            
            return points.astype(np.float32)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero point cloud if loading fails
            return np.zeros((self.num_points, 3), dtype=np.float32)
    
    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud."""
        if not self.is_training:
            return points
        
        # Random rotation around z-axis
        angle = np.random.uniform(-0.1, 0.1)  # Small rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                   [sin_a, cos_a, 0],
                                   [0, 0, 1]])
        points = points @ rotation_matrix.T
        
        # Random translation
        translation = np.random.uniform(-0.2, 0.2, size=3)
        points += translation
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        points *= scale
        
        # Add small noise
        noise = np.random.normal(0, 0.002, points.shape)
        points += noise
        
        return points
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Load point cloud sequence
        point_clouds = []
        for i in range(self.sequence_length + self.prediction_length):
            file_path = self.point_cloud_files[start_idx + i]
            pc = self._load_point_cloud(file_path)
            pc = self._augment_point_cloud(pc)
            point_clouds.append(pc)
        
        point_clouds = np.array(point_clouds)  # [seq_len + pred_len, num_points, 3]
        
        # Split into input and target
        input_sequence = point_clouds[:self.sequence_length]  # [seq_len, num_points, 3]
        target_sequence = point_clouds[self.sequence_length:]  # [pred_len, num_points, 3]
        
        return {
            'input_sequence': torch.FloatTensor(input_sequence),
            'target_sequence': torch.FloatTensor(target_sequence),
            'sequence_id': self.sequence_info[start_idx]
        }


def create_kitti_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for KITTI dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Training dataset (sequences 1-5)
    train_dataset = KITTIDataset(
        data_path=config['data']['data_path'],
        sequences=config['data']['train_sequences'],
        sequence_length=config['data']['sequence_length'],
        prediction_length=config['data']['prediction_length'],
        num_points=config['data']['num_points'],
        max_sequence_length=config['data']['max_sequence_length'],
        is_training=True
    )
    
    # Validation dataset (sequences 6-7)
    val_dataset = KITTIDataset(
        data_path=config['data']['data_path'],
        sequences=config['data']['val_sequences'],
        sequence_length=config['data']['sequence_length'],
        prediction_length=config['data']['prediction_length'],
        num_points=config['data']['num_points'],
        max_sequence_length=config['data']['max_sequence_length'],
        is_training=False
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test the dataloader
    import yaml
    
    with open('/home/pratyush/ISyE_Research/S2Net/configs/kitti_two_stream_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader = create_kitti_dataloaders(config)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Input shape: {batch['input_sequence'].shape}")
        print(f"Target shape: {batch['target_sequence'].shape}")
        print(f"Sequence ID: {batch['sequence_id']}")
        break


