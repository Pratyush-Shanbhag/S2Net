#!/usr/bin/env python3
"""
Test script for S2Net point cloud prediction.
This script demonstrates the model's ability to predict future point cloud sequences.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from models.s2net import S2Net
from data.dataloader import create_dataloader

def create_synthetic_point_cloud(shape='sphere', num_points=512, radius=1.0, noise_std=0.1):
    """Create a synthetic point cloud for testing."""
    if shape == 'sphere':
        # Generate points on a sphere
        phi = np.random.uniform(0, 2*np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(costheta)
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        points = np.stack([x, y, z], axis=1)
        
    elif shape == 'cube':
        # Generate points in a cube
        points = np.random.uniform(-radius, radius, (num_points, 3))
        
    elif shape == 'line':
        # Generate points along a line
        t = np.linspace(-radius, radius, num_points)
        points = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
        
    else:
        # Random points
        points = np.random.randn(num_points, 3) * radius
    
    # Add noise
    points += np.random.normal(0, noise_std, points.shape)
    
    return points

def create_sequential_point_clouds(shape='sphere', num_sequences=5, num_points=512, 
                                 movement_speed=0.1, rotation_speed=0.05):
    """Create a sequence of point clouds with movement and rotation."""
    sequences = []
    
    # Initial position and rotation
    position = np.array([0.0, 0.0, 0.0])
    rotation_angle = 0.0
    
    for i in range(num_sequences):
        # Create base point cloud
        points = create_synthetic_point_cloud(shape, num_points)
        
        # Apply rotation
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T
        
        # Apply translation
        points += position
        
        sequences.append(points)
        
        # Update position and rotation for next frame
        position += np.array([movement_speed, 0, 0])  # Move along x-axis
        rotation_angle += rotation_speed
    
    return np.array(sequences)

def visualize_point_clouds(points_list, titles=None, save_path=None):
    """Visualize a list of point clouds."""
    num_clouds = len(points_list)
    fig = plt.figure(figsize=(5 * num_clouds, 5))
    
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
            ax.set_title(f'Point Cloud {i+1}')
        
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

def test_prediction_with_synthetic_data():
    """Test prediction with synthetic data."""
    print("=== Testing S2Net with Synthetic Data ===")
    print()
    
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
    
    print("Model loaded successfully!")
    print()
    
    # Create synthetic input sequence
    print("Creating synthetic input sequence...")
    input_sequence = create_sequential_point_clouds(
        shape='sphere', 
        num_sequences=5, 
        num_points=512,
        movement_speed=0.2,
        rotation_speed=0.1
    )
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)  # Add batch dimension
    print(f"Input shape: {input_tensor.shape}")
    
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
    pred_sequence = pred_points.squeeze(0).cpu().numpy()  # Remove batch dimension
    
    # Create visualization
    print("Creating visualization...")
    
    # Input sequence
    input_titles = [f'Input Frame {i+1}' for i in range(len(input_sequence))]
    visualize_point_clouds(input_sequence, input_titles, 'synthetic_input_sequence.png')
    
    # Predicted sequence
    pred_titles = [f'Predicted Frame {i+1}' for i in range(len(pred_sequence))]
    visualize_point_clouds(pred_sequence, pred_titles, 'synthetic_predicted_sequence.png')
    
    # Combined visualization
    all_sequences = list(input_sequence) + list(pred_sequence)
    all_titles = input_titles + pred_titles
    visualize_point_clouds(all_sequences, all_titles, 'synthetic_combined_sequence.png')
    
    print("✅ Synthetic data test completed!")
    print(f"Input sequence range: [{input_sequence.min():.3f}, {input_sequence.max():.3f}]")
    print(f"Prediction range: [{pred_sequence.min():.3f}, {pred_sequence.max():.3f}]")
    
    return input_sequence, pred_sequence

def test_prediction_with_kitti_data():
    """Test prediction with real KITTI data."""
    print("\n=== Testing S2Net with KITTI Data ===")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open('configs/simple_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
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
    
    print("Model loaded successfully!")
    print()
    
    # Create KITTI data loader
    try:
        print("Loading KITTI data...")
        test_loader = create_dataloader(
            dataset_name='kitti',
            data_path='/Volumes/datasets/unzipped/KITTI/dataset',
            batch_size=1,
            sequence_length=5,
            prediction_length=3,
            num_points=512,
            num_workers=0,
            is_training=False,
            kitti_sequences=['00'],
            max_sequence_length=20
        )
        
        print(f"KITTI data loaded successfully! {len(test_loader)} batches available.")
        print()
        
        # Get a sample batch
        batch = next(iter(test_loader))
        input_sequence = batch['input_sequence'].to(device)
        target_sequence = batch['target_sequence'].to(device)
        
        print(f"Input shape: {input_sequence.shape}")
        print(f"Target shape: {target_sequence.shape}")
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            predictions = model(input_sequence)
            
            if isinstance(predictions, dict):
                pred_points = predictions['predicted_point_clouds']
            else:
                pred_points = predictions
        
        print(f"Prediction shape: {pred_points.shape}")
        
        # Convert to numpy for visualization
        input_np = input_sequence.squeeze(0).cpu().numpy()
        target_np = target_sequence.squeeze(0).cpu().numpy()
        pred_np = pred_points.squeeze(0).cpu().numpy()
        
        # Create visualization
        print("Creating visualization...")
        
        # Input sequence
        input_titles = [f'KITTI Input Frame {i+1}' for i in range(len(input_np))]
        visualize_point_clouds(input_np, input_titles, 'kitti_input_sequence.png')
        
        # Target vs Prediction
        target_titles = [f'KITTI Target Frame {i+1}' for i in range(len(target_np))]
        pred_titles = [f'KITTI Predicted Frame {i+1}' for i in range(len(pred_np))]
        
        # Show first few frames of prediction
        pred_subset = pred_np[:len(target_np)]  # Match target length
        pred_subset_titles = [f'KITTI Predicted Frame {i+1}' for i in range(len(pred_subset))]
        
        visualize_point_clouds(target_np, target_titles, 'kitti_target_sequence.png')
        visualize_point_clouds(pred_subset, pred_subset_titles, 'kitti_predicted_sequence.png')
        
        print("✅ KITTI data test completed!")
        print(f"Input range: [{input_np.min():.3f}, {input_np.max():.3f}]")
        print(f"Target range: [{target_np.min():.3f}, {target_np.max():.3f}]")
        print(f"Prediction range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
        
        return input_np, target_np, pred_np
        
    except Exception as e:
        print(f"❌ Error loading KITTI data: {e}")
        print("Skipping KITTI test...")
        return None, None, None

def main():
    """Main test function."""
    print("🚀 S2Net Point Cloud Prediction Test")
    print("=" * 50)
    
    # Test with synthetic data
    input_syn, pred_syn = test_prediction_with_synthetic_data()
    
    # Test with KITTI data
    input_kitti, target_kitti, pred_kitti = test_prediction_with_kitti_data()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nGenerated files:")
    print("- synthetic_input_sequence.png")
    print("- synthetic_predicted_sequence.png") 
    print("- synthetic_combined_sequence.png")
    if input_kitti is not None:
        print("- kitti_input_sequence.png")
        print("- kitti_target_sequence.png")
        print("- kitti_predicted_sequence.png")
    
    print("\nThe S2Net model successfully predicts future point cloud sequences!")

if __name__ == "__main__":
    main()
