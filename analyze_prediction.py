#!/usr/bin/env python3
"""
Analyze the prediction results for the specific KITTI sequence.
This script provides detailed analysis of the prediction quality and characteristics.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from models.s2net import S2Net

def load_kitti_bin_file(file_path, num_points=512):
    """Load a KITTI .bin file and return point cloud data."""
    try:
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]  # Extract x, y, z coordinates
        
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            padding = np.zeros((num_points - len(points), 3))
            points = np.vstack([points, padding])
        
        return points
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def create_sequence_from_file(file_path, sequence_length=5, num_points=512):
    """Create a sequence by loading multiple consecutive files."""
    dir_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_num = int(base_name.split('.')[0])
    
    sequence = []
    file_paths = []
    
    for i in range(sequence_length):
        target_file_num = file_num + i
        target_file = os.path.join(dir_path, f"{target_file_num:06d}.bin")
        file_paths.append(target_file)
        
        if os.path.exists(target_file):
            points = load_kitti_bin_file(target_file, num_points)
            if points is not None:
                sequence.append(points)
            else:
                if sequence:
                    sequence.append(sequence[-1].copy())
                else:
                    sequence.append(np.random.randn(num_points, 3) * 0.1)
        else:
            if sequence:
                sequence.append(sequence[-1].copy())
            else:
                sequence.append(np.random.randn(num_points, 3) * 0.1)
    
    return np.array(sequence), file_paths

def analyze_point_cloud_sequence(sequence, title="Point Cloud Analysis"):
    """Analyze a point cloud sequence and return statistics."""
    print(f"\n📊 {title}")
    print("-" * 40)
    
    # Basic statistics
    print(f"Sequence shape: {sequence.shape}")
    print(f"Value range: [{sequence.min():.3f}, {sequence.max():.3f}]")
    print(f"Mean: {sequence.mean():.3f}")
    print(f"Std: {sequence.std():.3f}")
    
    # Per-frame analysis
    print(f"\nPer-frame analysis:")
    for i in range(len(sequence)):
        frame = sequence[i]
        centroid = np.mean(frame, axis=0)
        spread = np.std(frame, axis=0)
        print(f"  Frame {i+1}: centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
              f"spread=({spread[0]:.2f}, {spread[1]:.2f}, {spread[2]:.2f})")
    
    # Temporal analysis
    if len(sequence) > 1:
        movements = []
        for i in range(1, len(sequence)):
            movement = np.linalg.norm(sequence[i] - sequence[i-1], axis=1).mean()
            movements.append(movement)
        
        print(f"\nTemporal movement (avg point displacement):")
        for i, movement in enumerate(movements):
            print(f"  Frame {i+1} -> {i+2}: {movement:.3f}")
        print(f"  Average movement: {np.mean(movements):.3f}")
        print(f"  Movement std: {np.std(movements):.3f}")
        
        # Centroid movement
        centroids = np.mean(sequence, axis=1)
        centroid_movements = []
        for i in range(1, len(centroids)):
            centroid_movement = np.linalg.norm(centroids[i] - centroids[i-1])
            centroid_movements.append(centroid_movement)
        
        print(f"\nCentroid movement:")
        for i, movement in enumerate(centroid_movements):
            print(f"  Frame {i+1} -> {i+2}: {movement:.3f}")
        print(f"  Average centroid movement: {np.mean(centroid_movements):.3f}")

def compare_sequences(input_seq, pred_seq):
    """Compare input and predicted sequences."""
    print(f"\n🔍 Sequence Comparison")
    print("=" * 50)
    
    # Basic comparison
    print(f"Input shape: {input_seq.shape}")
    print(f"Prediction shape: {pred_seq.shape}")
    print(f"Input range: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
    print(f"Prediction range: [{pred_seq.min():.3f}, {pred_seq.max():.3f}]")
    
    # Statistical comparison
    print(f"\nStatistical comparison:")
    print(f"Input mean: {input_seq.mean():.3f}, std: {input_seq.std():.3f}")
    print(f"Prediction mean: {pred_seq.mean():.3f}, std: {pred_seq.std():.3f}")
    
    # Temporal consistency comparison
    if len(input_seq) > 1 and len(pred_seq) > 1:
        input_movements = []
        for i in range(1, len(input_seq)):
            movement = np.linalg.norm(input_seq[i] - input_seq[i-1], axis=1).mean()
            input_movements.append(movement)
        
        pred_movements = []
        for i in range(1, len(pred_seq)):
            movement = np.linalg.norm(pred_seq[i] - pred_seq[i-1], axis=1).mean()
            pred_movements.append(movement)
        
        print(f"\nTemporal movement comparison:")
        print(f"Input average movement: {np.mean(input_movements):.3f}")
        print(f"Prediction average movement: {np.mean(pred_movements):.3f}")
        print(f"Movement ratio (pred/input): {np.mean(pred_movements)/np.mean(input_movements):.3f}")

def main():
    """Main analysis function."""
    print("🔬 S2Net Prediction Analysis for KITTI Sequence 000653")
    print("=" * 60)
    
    # Target file
    target_file = "/Volumes/datasets/unzipped/KITTI/dataset/sequences/01/velodyne/000653.bin"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration and model
    with open('configs/simple_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    model = S2Net(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_points=model_config['num_points'],
        num_lstm_layers=model_config['num_lstm_layers'],
        num_pyramid_levels=model_config['num_pyramid_levels'],
        use_temporal_variational=model_config['use_temporal_variational'],
        use_multi_scale=model_config['use_multi_scale'],
        use_uncertainty=model_config['use_uncertainty'],
        use_temporal_decoder=model_config['use_temporal_decoder'],
        dropout=model_config['dropout']
    )
    
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load sequence
    print(f"\nLoading sequence from: {target_file}")
    input_sequence, file_paths = create_sequence_from_file(target_file, sequence_length=5, num_points=512)
    
    print(f"Loaded files:")
    for i, path in enumerate(file_paths):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  Frame {i+1}: {os.path.basename(path)} {exists}")
    
    # Make prediction
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(input_tensor)
        if isinstance(predictions, dict):
            pred_points = predictions['predicted_point_clouds']
        else:
            pred_points = predictions
    
    pred_sequence = pred_points.squeeze(0).cpu().numpy()
    
    # Analyze sequences
    analyze_point_cloud_sequence(input_sequence, "Input Sequence Analysis")
    analyze_point_cloud_sequence(pred_sequence, "Predicted Sequence Analysis")
    compare_sequences(input_sequence, pred_sequence)
    
    print(f"\n✅ Analysis completed!")
    print(f"The model successfully predicted future point cloud sequences for KITTI file 000653.bin")

if __name__ == "__main__":
    main()
