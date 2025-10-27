#!/usr/bin/env python3
"""
Test script for S2Net Two-Stream model on NuScenes test dataset.
Tests on the first half of the test dataset as requested.
"""

import os
import sys
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import time

# Add project root to path
sys.path.append('/home/pratyush/ISyE_Research/S2Net')

from models.s2net_two_stream import TwoStreamS2Net
from utils.improved_losses import ImprovedChamferDistanceLoss
from utils.metrics import compute_chamfer_distance, compute_hausdorff_distance


class NuScenesTestDataset(Dataset):
    """Dataset for NuScenes test data."""
    
    def __init__(self, 
                 lidar_dir: str,
                 metadata_dir: str,
                 num_files: int = 3004,  # First half of test dataset
                 sequence_length: int = 5,
                 prediction_length: int = 3,
                 num_points: int = 512):
        """
        Initialize NuScenes test dataset.
        
        Args:
            lidar_dir: Path to LIDAR_TOP directory
            metadata_dir: Path to metadata directory
            num_files: Number of files to use (first half)
            sequence_length: Input sequence length
            prediction_length: Future prediction length
            num_points: Number of points per point cloud
        """
        self.lidar_dir = lidar_dir
        self.metadata_dir = metadata_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_points = num_points
        
        # Get all LiDAR files and sort them
        self.lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd.bin")))
        
        # Use only first half of the dataset
        self.lidar_files = self.lidar_files[:num_files]
        
        # Load metadata
        self.samples = self._load_samples()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences from {len(self.lidar_files)} files")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata."""
        sample_file = os.path.join(self.metadata_dir, "sample.json")
        with open(sample_file, 'r') as f:
            samples = json.load(f)
        return samples
    
    def _create_sequences(self) -> List[List[str]]:
        """Create sequences from consecutive samples."""
        sequences = []
        
        # Group files by scene (using filename pattern)
        scene_groups = {}
        for file_path in self.lidar_files:
            filename = os.path.basename(file_path)
            # Extract scene identifier from filename
            scene_id = filename.split('__')[0]  # e.g., "n008-2018-08-01-15-34-25-0400"
            
            if scene_id not in scene_groups:
                scene_groups[scene_id] = []
            scene_groups[scene_id].append(file_path)
        
        # Create sequences from each scene
        for scene_id, files in scene_groups.items():
            # Sort files by timestamp (filename contains timestamp)
            files.sort()
            
            # Create sequences of length sequence_length + prediction_length
            for i in range(len(files) - self.sequence_length - self.prediction_length + 1):
                sequence = files[i:i + self.sequence_length + self.prediction_length]
                sequences.append(sequence)
        
        return sequences
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from .pcd.bin file."""
        try:
            # Read binary file
            points = np.fromfile(file_path, dtype=np.float32)
            
            # Reshape to (N, 4) - x, y, z, intensity
            points = points.reshape(-1, 4)
            
            # Take only x, y, z coordinates
            points = points[:, :3]
            
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
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_files = self.sequences[idx]
        
        # Load point clouds
        point_clouds = []
        for file_path in sequence_files:
            pc = self._load_point_cloud(file_path)
            point_clouds.append(pc)
        
        point_clouds = np.array(point_clouds)  # [seq_len + pred_len, num_points, 3]
        
        # Split into input and target
        input_sequence = point_clouds[:self.sequence_length]  # [seq_len, num_points, 3]
        target_sequence = point_clouds[self.sequence_length:]  # [pred_len, num_points, 3]
        
        return {
            'input_sequence': torch.FloatTensor(input_sequence),
            'target_sequence': torch.FloatTensor(target_sequence),
            'sequence_files': sequence_files
        }


def load_model(checkpoint_path: str, device: str = 'cuda') -> TwoStreamS2Net:
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with parameters from checkpoint
    model_config = checkpoint.get('model_config', {
        'input_dim': 3,
        'hidden_dim': 256,
        'latent_dim': 128,
        'num_points': 512,
        'use_temporal_variational': True,
        'use_multi_scale': False
    })
    
    model = TwoStreamS2Net(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {model.get_model_size():,}")
    return model


def evaluate_model(model: TwoStreamS2Net, 
                   dataloader: DataLoader,
                   device: str = 'cuda',
                   num_samples: int = 1) -> Dict[str, float]:
    """Evaluate the model on the test dataset."""
    
    model.eval()
    
    # Initialize metrics
    total_chamfer_det = 0.0
    total_chamfer_stoch = 0.0
    total_hausdorff_det = 0.0
    total_hausdorff_stoch = 0.0
    total_sequences = 0
    
    # Loss functions
    chamfer_loss = ImprovedChamferDistanceLoss()
    
    print("Evaluating model on NuScenes test dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            input_seq = batch['input_sequence'].to(device)  # [batch_size, seq_len, num_points, 3]
            target_seq = batch['target_sequence'].to(device)  # [batch_size, pred_len, num_points, 3]
            
            batch_size = input_seq.shape[0]
            
            # Forward pass
            predictions = model(input_seq, is_training=False)
            
            # Get predictions (TwoStreamS2Net returns deterministic and stochastic predictions)
            det_pred = predictions['deterministic_predictions']  # [batch_size, seq_len, num_points, 3]
            stoch_pred = predictions['stochastic_predictions']  # [batch_size, seq_len, num_points, 3]
            
            # Compute metrics for each sequence in batch
            for i in range(batch_size):
                # Deterministic stream metrics
                det_chamfer = compute_chamfer_distance(
                    det_pred[i].cpu().numpy(), 
                    target_seq[i].cpu().numpy()
                )
                det_hausdorff = compute_hausdorff_distance(
                    det_pred[i].cpu().numpy(), 
                    target_seq[i].cpu().numpy()
                )
                
                # Stochastic stream metrics
                stoch_chamfer = compute_chamfer_distance(
                    stoch_pred[i].cpu().numpy(), 
                    target_seq[i].cpu().numpy()
                )
                stoch_hausdorff = compute_hausdorff_distance(
                    stoch_pred[i].cpu().numpy(), 
                    target_seq[i].cpu().numpy()
                )
                
                total_chamfer_det += det_chamfer
                total_chamfer_stoch += stoch_chamfer
                total_hausdorff_det += det_hausdorff
                total_hausdorff_stoch += stoch_hausdorff
                total_sequences += 1
    
    # Compute average metrics
    avg_chamfer_det = total_chamfer_det / total_sequences
    avg_chamfer_stoch = total_chamfer_stoch / total_sequences
    avg_hausdorff_det = total_hausdorff_det / total_sequences
    avg_hausdorff_stoch = total_hausdorff_stoch / total_sequences
    
    results = {
        'chamfer_distance_deterministic': avg_chamfer_det,
        'chamfer_distance_stochastic': avg_chamfer_stoch,
        'hausdorff_distance_deterministic': avg_hausdorff_det,
        'hausdorff_distance_stochastic': avg_hausdorff_stoch,
        'total_sequences': total_sequences
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test S2Net on NuScenes test dataset')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/pratyush/ISyE_Research/S2Net/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--lidar_dir', type=str,
                       default='/home/pratyush/ISyE_Research/datasets/unzipped/Nuscenes/v1.0-test_blobs/samples/LIDAR_TOP',
                       help='Path to LIDAR_TOP directory')
    parser.add_argument('--metadata_dir', type=str,
                       default='/home/pratyush/ISyE_Research/datasets/unzipped/Nuscenes/v1.0-test_meta/v1.0-test',
                       help='Path to metadata directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    parser.add_argument('--num_files', type=int, default=3004,
                       help='Number of files to use (first half of test dataset)')
    parser.add_argument('--sequence_length', type=int, default=5,
                       help='Input sequence length')
    parser.add_argument('--prediction_length', type=int, default=3,
                       help='Future prediction length')
    parser.add_argument('--num_points', type=int, default=512,
                       help='Number of points per point cloud')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating NuScenes test dataset...")
    dataset = NuScenesTestDataset(
        lidar_dir=args.lidar_dir,
        metadata_dir=args.metadata_dir,
        num_files=args.num_files,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        num_points=args.num_points
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    model = load_model(args.checkpoint, device)
    model = model.to(device)
    
    # Evaluate model
    print("Starting evaluation...")
    start_time = time.time()
    
    results = evaluate_model(model, dataloader, device)
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Print results
    print("\n" + "="*60)
    print("NUSCENES TEST DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: NuScenes Test (First Half)")
    print(f"Total sequences evaluated: {results['total_sequences']}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Time per sequence: {evaluation_time/results['total_sequences']:.4f} seconds")
    print("\nDETERMINISTIC STREAM:")
    print(f"  Chamfer Distance: {results['chamfer_distance_deterministic']:.6f}")
    print(f"  Hausdorff Distance: {results['hausdorff_distance_deterministic']:.6f}")
    print("\nSTOCHASTIC STREAM:")
    print(f"  Chamfer Distance: {results['chamfer_distance_stochastic']:.6f}")
    print(f"  Hausdorff Distance: {results['hausdorff_distance_stochastic']:.6f}")
    print("\nOVERALL PERFORMANCE:")
    avg_chamfer = (results['chamfer_distance_deterministic'] + results['chamfer_distance_stochastic']) / 2
    avg_hausdorff = (results['hausdorff_distance_deterministic'] + results['hausdorff_distance_stochastic']) / 2
    print(f"  Average Chamfer Distance: {avg_chamfer:.6f}")
    print(f"  Average Hausdorff Distance: {avg_hausdorff:.6f}")
    print("="*60)
    
    # Save results
    results_file = '/home/pratyush/ISyE_Research/S2Net/nuscenes_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
