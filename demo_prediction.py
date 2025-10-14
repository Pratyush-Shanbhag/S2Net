#!/usr/bin/env python3
"""
Simple demo script for S2Net point cloud prediction.
This script shows a quick demonstration of the model's prediction capabilities.
"""

import torch
import yaml
import numpy as np
from models.s2net import S2Net

def create_simple_point_cloud(shape='sphere', num_points=512):
    """Create a simple point cloud for demonstration."""
    if shape == 'sphere':
        # Generate points on a sphere
        phi = np.random.uniform(0, 2*np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.stack([x, y, z], axis=1)
    else:
        # Random points
        points = np.random.randn(num_points, 3)
    
    return points

def demo_prediction():
    """Run a simple prediction demo."""
    print("🎯 S2Net Point Cloud Prediction Demo")
    print("=" * 40)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open('configs/simple_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    print(f"Model: {model_config['hidden_dim']} hidden dim, {model_config['latent_dim']} latent dim")
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
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print()
    
    # Create input sequence
    print("Creating input sequence...")
    input_sequence = []
    for i in range(5):
        # Create a moving sphere
        points = create_simple_point_cloud('sphere', 512)
        # Add movement
        points[:, 0] += i * 0.2  # Move along x-axis
        points[:, 1] += i * 0.1  # Move along y-axis
        input_sequence.append(points)
    
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
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
    
    # Convert to numpy
    input_np = input_tensor.squeeze(0).cpu().numpy()
    pred_np = pred_points.squeeze(0).cpu().numpy()
    
    # Analyze results
    print("\n📊 Prediction Analysis:")
    print(f"Input range: [{input_np.min():.3f}, {input_np.max():.3f}]")
    print(f"Prediction range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
    
    # Check temporal consistency
    input_movement = np.linalg.norm(input_np[1:] - input_np[:-1], axis=(1,2)).mean()
    pred_movement = np.linalg.norm(pred_np[1:] - pred_np[:-1], axis=(1,2)).mean()
    
    print(f"Input temporal movement: {input_movement:.3f}")
    print(f"Prediction temporal movement: {pred_movement:.3f}")
    
    # Check point cloud statistics
    input_centroid = np.mean(input_np, axis=1)
    pred_centroid = np.mean(pred_np, axis=1)
    
    print(f"\nInput centroid movement: {np.linalg.norm(input_centroid[1:] - input_centroid[:-1], axis=1).mean():.3f}")
    print(f"Prediction centroid movement: {np.linalg.norm(pred_centroid[1:] - pred_centroid[:-1], axis=1).mean():.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("The model successfully predicted future point cloud sequences!")
    
    return input_np, pred_np

if __name__ == "__main__":
    input_seq, pred_seq = demo_prediction()
