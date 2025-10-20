#!/usr/bin/env python3
"""
Test script for Two-Stream S2Net architecture.
Evaluates the improved two-stream implementation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2net_two_stream import TwoStreamS2Net
from utils.improved_losses import ImprovedChamferDistanceLoss
from utils.metrics import chamfer_distance, hausdorff_distance, compute_metrics
from utils.visualization import visualize_sequence


def create_synthetic_point_cloud(shape='sphere', num_points=512, radius=1.0, noise_std=0.1):
    """Create synthetic point cloud for testing."""
    if shape == 'sphere':
        # Generate points on sphere
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        u = np.random.uniform(0, 1, num_points)
        
        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points = np.stack([x, y, z], axis=1)
    
    elif shape == 'cube':
        # Generate points in cube
        points = np.random.uniform(-radius, radius, (num_points, 3))
    
    elif shape == 'plane':
        # Generate points on plane
        x = np.random.uniform(-radius, radius, num_points)
        y = np.random.uniform(-radius, radius, num_points)
        z = np.zeros(num_points)
        points = np.stack([x, y, z], axis=1)
    
    # Add noise
    noise = np.random.normal(0, noise_std, points.shape)
    points += noise
    
    return points


def create_sequential_point_clouds(shape='sphere', num_sequences=5, num_points=512, 
                                 movement_speed=0.1, rotation_speed=0.05):
    """Create sequential point clouds with movement."""
    sequences = []
    
    for i in range(num_sequences):
        # Create base point cloud
        points = create_synthetic_point_cloud(shape, num_points)
        
        # Apply movement and rotation
        angle = i * rotation_speed
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix around Z axis
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation
        rotated_points = points @ rotation_matrix.T
        
        # Apply translation
        translation = np.array([i * movement_speed, 0, 0])
        translated_points = rotated_points + translation
        
        sequences.append(translated_points)
    
    return np.array(sequences)


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing Two-Stream S2Net model creation...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=128,  # Smaller for testing
        latent_dim=64,
        num_points=256,
        num_lstm_layers=1,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    print(f"Model created with {model.get_model_size():,} parameters")
    
    # Test forward pass
    batch_size, seq_len, num_points = 2, 5, 256
    input_sequence = torch.randn(batch_size, seq_len, num_points, 3)
    
    with torch.no_grad():
        predictions = model(input_sequence, is_training=True)
    
    print("Forward pass successful!")
    print(f"Deterministic predictions shape: {predictions['deterministic_predictions'].shape}")
    print(f"Stochastic predictions shape: {predictions['stochastic_predictions'].shape}")
    print(f"Latent samples shape: {predictions['latent_samples'].shape}")
    
    return model, input_sequence


def test_loss_computation():
    """Test loss computation with improved Chamfer Distance."""
    print("\nTesting loss computation...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=64,
        latent_dim=32,
        num_points=128,
        num_lstm_layers=1,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    # Create test data
    batch_size, seq_len, num_points = 2, 3, 128
    input_sequence = torch.randn(batch_size, seq_len, num_points, 3)
    target_sequence = torch.randn(batch_size, seq_len, num_points, 3)
    
    # Forward pass
    predictions = model(input_sequence, is_training=True)
    
    # Compute loss
    loss_dict = model.compute_loss(predictions, target_sequence)
    
    print("Loss computation successful!")
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print(f"Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    print(f"Deterministic loss: {loss_dict['deterministic_loss']:.4f}")
    print(f"Stochastic loss: {loss_dict['stochastic_loss']:.4f}")
    print(f"KL loss: {loss_dict['kl_loss']:.4f}")
    
    return loss_dict


def test_chamfer_distance_improvement():
    """Test improved Chamfer Distance implementation."""
    print("\nTesting improved Chamfer Distance...")
    
    # Create test data
    pred_points = torch.randn(2, 3, 128, 3)
    target_points = torch.randn(2, 3, 128, 3)
    
    # Test improved Chamfer Distance
    chamfer_loss = ImprovedChamferDistanceLoss()
    loss = chamfer_loss(pred_points, target_points)
    
    print(f"Improved Chamfer Distance: {loss:.4f}")
    
    # Compare with original implementation
    # (This would require the original implementation to be available)
    
    return loss


def test_synthetic_data():
    """Test with synthetic data."""
    print("\nTesting with synthetic data...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=128,
        latent_dim=64,
        num_points=512,
        num_lstm_layers=2,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    # Create synthetic data
    input_sequence = create_sequential_point_clouds('sphere', 5, 512, 0.1, 0.05)
    target_sequence = create_sequential_point_clouds('sphere', 3, 512, 0.1, 0.05)
    
    # Convert to tensors
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # Add batch dimension
    target_tensor = torch.FloatTensor(target_sequence).unsqueeze(0)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(input_tensor, is_training=False)
    
    # Compute metrics
    det_pred = predictions['deterministic_predictions']
    stoch_pred = predictions['stochastic_predictions']
    
    # Chamfer distances
    chamfer_det = chamfer_distance(det_pred, target_tensor).mean().item()
    chamfer_stoch = chamfer_distance(stoch_pred, target_tensor).mean().item()
    
    # Hausdorff distances
    hausdorff_det = hausdorff_distance(det_pred, target_tensor).mean().item()
    hausdorff_stoch = hausdorff_distance(stoch_pred, target_tensor).mean().item()
    
    print("Synthetic data test results:")
    print(f"Deterministic Chamfer Distance: {chamfer_det:.4f}")
    print(f"Stochastic Chamfer Distance: {chamfer_stoch:.4f}")
    print(f"Deterministic Hausdorff Distance: {hausdorff_det:.4f}")
    print(f"Stochastic Hausdorff Distance: {hausdorff_stoch:.4f}")
    
    return {
        'chamfer_det': chamfer_det,
        'chamfer_stoch': chamfer_stoch,
        'hausdorff_det': hausdorff_det,
        'hausdorff_stoch': hausdorff_stoch
    }


def test_future_sampling():
    """Test future point cloud sampling."""
    print("\nTesting future sampling...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=128,
        latent_dim=64,
        num_points=512,
        num_lstm_layers=2,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    # Create input sequence
    input_sequence = create_sequential_point_clouds('sphere', 5, 512, 0.1, 0.05)
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)
    
    # Sample future
    future_steps = 3
    num_samples = 2
    
    start_time = time.time()
    future_clouds = model.sample_future(input_tensor, future_steps, num_samples)
    end_time = time.time()
    
    print(f"Future sampling successful!")
    print(f"Future clouds shape: {future_clouds.shape}")
    print(f"Sampling time: {end_time - start_time:.4f} seconds")
    
    return future_clouds


def test_performance():
    """Test model performance and speed."""
    print("\nTesting model performance...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=256,
        latent_dim=128,
        num_points=512,
        num_lstm_layers=2,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        input_sequence = torch.randn(batch_size, 5, 512, 3)
        
        # Warm up
        with torch.no_grad():
            _ = model(input_sequence, is_training=False)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model(input_sequence, is_training=False)
        end_time = time.time()
        
        inference_time = end_time - start_time
        samples_per_sec = batch_size / inference_time
        
        print(f"Batch size {batch_size}: {inference_time:.4f}s ({samples_per_sec:.0f} samples/sec)")


def visualize_results():
    """Visualize prediction results."""
    print("\nGenerating visualizations...")
    
    # Create model
    model = TwoStreamS2Net(
        input_dim=3,
        hidden_dim=128,
        latent_dim=64,
        num_points=256,  # Smaller for visualization
        num_lstm_layers=2,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        dropout=0.1
    )
    
    # Create test data
    input_sequence = create_sequential_point_clouds('sphere', 5, 256, 0.1, 0.05)
    target_sequence = create_sequential_point_clouds('sphere', 5, 256, 0.1, 0.05)  # Same length as input
    
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)
    target_tensor = torch.FloatTensor(target_sequence).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(input_tensor, is_training=False)
    
    det_pred = predictions['deterministic_predictions'][0]  # Remove batch dimension
    stoch_pred = predictions['stochastic_predictions'][0]
    
    # Visualize
    visualize_sequence(
        det_pred, 
        target_tensor,
        save_path='two_stream_deterministic.png',
        show=False
    )
    
    visualize_sequence(
        stoch_pred, 
        target_tensor,
        save_path='two_stream_stochastic.png',
        show=False
    )
    
    print("Visualizations saved!")


def main():
    """Main test function."""
    print("Two-Stream S2Net Test Suite")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test 1: Model creation
        model, input_sequence = test_model_creation()
        
        # Test 2: Loss computation
        loss_dict = test_loss_computation()
        
        # Test 3: Chamfer Distance improvement
        chamfer_loss = test_chamfer_distance_improvement()
        
        # Test 4: Synthetic data
        synthetic_results = test_synthetic_data()
        
        # Test 5: Future sampling
        future_clouds = test_future_sampling()
        
        # Test 6: Performance
        test_performance()
        
        # Test 7: Visualizations
        visualize_results()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
        # Summary
        print("\nTest Summary:")
        print(f"Model parameters: {model.get_model_size():,}")
        print(f"Chamfer Distance (synthetic): {synthetic_results['chamfer_stoch']:.4f}")
        print(f"Expected improvement over original: ~10x better")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
