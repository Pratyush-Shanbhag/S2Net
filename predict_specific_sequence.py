#!/usr/bin/env python3
"""
Predict point cloud for a specific KITTI sequence file.
This script loads a specific .bin file and generates predictions using the trained S2Net model.
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
        # Load binary data
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
        # Extract x, y, z coordinates (ignore intensity)
        points = points[:, :3]
        
        # Subsample to desired number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            # Pad with zeros if not enough points
            padding = np.zeros((num_points - len(points), 3))
            points = np.vstack([points, padding])
        
        return points
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def create_sequence_from_file(file_path, sequence_length=5, num_points=512):
    """Create a sequence by loading multiple consecutive files."""
    # Extract directory and base filename
    dir_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_num = int(base_name.split('.')[0])
    
    sequence = []
    
    # Load the specified file and surrounding files
    for i in range(sequence_length):
        target_file_num = file_num + i
        target_file = os.path.join(dir_path, f"{target_file_num:06d}.bin")
        
        if os.path.exists(target_file):
            points = load_kitti_bin_file(target_file, num_points)
            if points is not None:
                sequence.append(points)
            else:
                # If file doesn't exist or failed to load, use the previous frame
                if sequence:
                    sequence.append(sequence[-1].copy())
                else:
                    # Create random points as fallback
                    sequence.append(np.random.randn(num_points, 3) * 0.1)
        else:
            # If file doesn't exist, use the previous frame or create random
            if sequence:
                sequence.append(sequence[-1].copy())
            else:
                sequence.append(np.random.randn(num_points, 3) * 0.1)
    
    return np.array(sequence)

def visualize_point_clouds(points_list, titles=None, save_path=None, figsize=(15, 5)):
    """Visualize a list of point clouds."""
    num_clouds = len(points_list)
    fig = plt.figure(figsize=figsize)
    
    for i, points in enumerate(points_list):
        ax = fig.add_subplot(1, num_clouds, i+1, projection='3d')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if titles:
            ax.set_title(titles[i])
        else:
            ax.set_title(f'Frame {i+1}')
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def predict_sequence(file_path):
    """Predict point cloud sequence for a specific KITTI file."""
    print("🎯 S2Net Point Cloud Prediction for Specific KITTI Sequence")
    print("=" * 60)
    print(f"Target file: {file_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None, None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open('configs/simple_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    print(f"Model configuration: {model_config}")
    print()
    
    # Create and load model
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
    
    # Load trained weights
    checkpoint_path = 'checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print()
    
    # Load the specific sequence
    print("Loading KITTI sequence...")
    input_sequence = create_sequence_from_file(file_path, sequence_length=5, num_points=512)
    
    if input_sequence is None:
        print("❌ Failed to load sequence")
        return None, None, None
    
    print(f"Input sequence shape: {input_sequence.shape}")
    print(f"Input range: [{input_sequence.min():.3f}, {input_sequence.max():.3f}]")
    print()
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        predictions = model(input_tensor)
        
        if isinstance(predictions, dict):
            pred_points = predictions['predicted_point_clouds']
        else:
            pred_points = predictions
    
    print(f"Prediction shape: {pred_points.shape}")
    
    # Convert back to numpy
    pred_sequence = pred_points.squeeze(0).cpu().numpy()
    
    print(f"Prediction range: [{pred_sequence.min():.3f}, {pred_sequence.max():.3f}]")
    print()
    
    # Analyze the prediction
    print("📊 Prediction Analysis:")
    
    # Check temporal consistency
    input_movement = np.linalg.norm(input_sequence[1:] - input_sequence[:-1], axis=(1,2)).mean()
    pred_movement = np.linalg.norm(pred_sequence[1:] - pred_sequence[:-1], axis=(1,2)).mean()
    
    print(f"Input temporal movement: {input_movement:.3f}")
    print(f"Prediction temporal movement: {pred_movement:.3f}")
    
    # Check point cloud statistics
    input_centroid = np.mean(input_sequence, axis=1)
    pred_centroid = np.mean(pred_sequence, axis=1)
    
    print(f"Input centroid movement: {np.linalg.norm(input_centroid[1:] - input_centroid[:-1], axis=1).mean():.3f}")
    print(f"Prediction centroid movement: {np.linalg.norm(pred_centroid[1:] - pred_centroid[:-1], axis=1).mean():.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Input sequence
    input_titles = [f'KITTI Input Frame {i+1}' for i in range(len(input_sequence))]
    visualize_point_clouds(input_sequence, input_titles, 'kitti_specific_input.png', figsize=(20, 4))
    
    # Predicted sequence
    pred_titles = [f'KITTI Predicted Frame {i+1}' for i in range(len(pred_sequence))]
    visualize_point_clouds(pred_sequence, pred_titles, 'kitti_specific_prediction.png', figsize=(20, 4))
    
    # Combined visualization
    all_sequences = list(input_sequence) + list(pred_sequence)
    all_titles = input_titles + pred_titles
    visualize_point_clouds(all_sequences, all_titles, 'kitti_specific_combined.png', figsize=(30, 4))
    
    print("✅ Prediction completed successfully!")
    print("\nGenerated files:")
    print("- kitti_specific_input.png")
    print("- kitti_specific_prediction.png")
    print("- kitti_specific_combined.png")
    
    return input_sequence, pred_sequence

def main():
    """Main function."""
    # Target file path
    target_file = "/Volumes/datasets/unzipped/KITTI/dataset/sequences/01/velodyne/000653.bin"
    
    # Run prediction
    input_seq, pred_seq = predict_sequence(target_file)
    
    if input_seq is not None:
        print("\n🎉 Point cloud prediction completed successfully!")
        print(f"The model has predicted future point cloud sequences for file: {target_file}")
    else:
        print("\n❌ Prediction failed. Please check the file path and try again.")

if __name__ == "__main__":
    main()
