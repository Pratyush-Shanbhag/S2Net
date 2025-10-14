#!/usr/bin/env python3
"""
Demo script for S2Net: Stochastic Sequential Pointcloud Forecasting

This script demonstrates how to use the S2Net model for point cloud sequence prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from models import S2Net
from data import create_dataloader
from utils.visualization import visualize_sequence, save_point_cloud_sequence


def create_demo_data(batch_size=2, seq_len=10, num_points=512):
    """
    Create synthetic demo data for testing.
    
    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        num_points: Number of points per point cloud
        
    Returns:
        input_sequence: Input point cloud sequences [batch_size, seq_len, num_points, 3]
        target_sequence: Target point cloud sequences [batch_size, seq_len, num_points, 3]
    """
    print("Creating demo data...")
    
    input_sequence = []
    target_sequence = []
    
    for b in range(batch_size):
        # Create a moving point cloud
        base_points = np.random.randn(num_points, 3) * 2
        
        input_seq = []
        target_seq = []
        
        for t in range(seq_len):
            # Add temporal motion
            time_offset = t * 0.1
            motion = np.array([
                np.sin(time_offset) * 0.5,
                np.cos(time_offset) * 0.5,
                np.sin(time_offset * 2) * 0.2
            ])
            
            # Create point cloud at time t
            points = base_points + motion
            input_seq.append(points)
            
            # Create target (future) point cloud
            future_time_offset = (t + seq_len) * 0.1
            future_motion = np.array([
                np.sin(future_time_offset) * 0.5,
                np.cos(future_time_offset) * 0.5,
                np.sin(future_time_offset * 2) * 0.2
            ])
            
            future_points = base_points + future_motion
            target_seq.append(future_points)
        
        input_sequence.append(np.stack(input_seq))
        target_sequence.append(np.stack(target_seq))
    
    input_sequence = torch.tensor(np.stack(input_sequence), dtype=torch.float32)
    target_sequence = torch.tensor(np.stack(target_sequence), dtype=torch.float32)
    
    print(f"Created demo data: {input_sequence.shape} input, {target_sequence.shape} target")
    return input_sequence, target_sequence


def demo_model_creation():
    """Demonstrate model creation and basic functionality."""
    print("\n" + "="*50)
    print("DEMO: Model Creation")
    print("="*50)
    
    # Create model
    model = S2Net(
        input_dim=3,
        hidden_dim=128,  # Smaller for demo
        latent_dim=64,
        num_points=512,
        num_lstm_layers=2,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=True,
        use_uncertainty=True,
        dropout=0.1
    )
    
    print(f"Model created with {model.get_model_size():,} parameters")
    
    # Test forward pass
    batch_size, seq_len, num_points = 2, 5, 512
    input_sequence, target_sequence = create_demo_data(batch_size, seq_len, num_points)
    
    print(f"Input shape: {input_sequence.shape}")
    print(f"Target shape: {target_sequence.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(input_sequence, is_training=False)
    
    print(f"Predicted shape: {predictions['predicted_point_clouds'].shape}")
    print(f"Latent samples shape: {predictions['latent_samples'].shape}")
    
    if 'uncertainties' in predictions:
        print(f"Uncertainties shape: {predictions['uncertainties'].shape}")
    
    return model, input_sequence, target_sequence, predictions


def demo_training():
    """Demonstrate training process."""
    print("\n" + "="*50)
    print("DEMO: Training Process")
    print("="*50)
    
    # Create model
    model = S2Net(
        input_dim=3,
        hidden_dim=64,  # Very small for demo
        latent_dim=32,
        num_points=256,
        num_lstm_layers=1,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,  # Disable for faster demo
        use_uncertainty=False,  # Disable for faster demo
        dropout=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create demo data
    input_sequence, target_sequence = create_demo_data(batch_size=4, seq_len=5, num_points=256)
    
    print("Training for 5 epochs...")
    
    # Training loop
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(input_sequence, is_training=True)
        
        # Compute loss
        loss_dict = model.compute_loss(predictions, target_sequence)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/5: Loss = {loss.item():.4f}, "
              f"Recon = {loss_dict['reconstruction_loss'].item():.4f}, "
              f"KL = {loss_dict['kl_loss'].item():.4f}")
    
    print("Training completed!")
    return model


def demo_prediction():
    """Demonstrate prediction capabilities."""
    print("\n" + "="*50)
    print("DEMO: Prediction")
    print("="*50)
    
    # Create trained model (simplified)
    model = S2Net(
        input_dim=3,
        hidden_dim=64,
        latent_dim=32,
        num_points=256,
        num_lstm_layers=1,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        use_uncertainty=False,
        dropout=0.1
    )
    
    # Create demo data
    input_sequence, target_sequence = create_demo_data(batch_size=1, seq_len=5, num_points=256)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        # Single prediction
        predictions = model(input_sequence, is_training=False)
        single_pred = predictions['predicted_point_clouds']
        
        # Multiple samples
        multi_pred = model.sample_future(
            input_sequence,
            future_steps=5,
            num_samples=3
        )
    
    print(f"Single prediction shape: {single_pred.shape}")
    print(f"Multiple samples shape: {multi_pred.shape}")
    
    # Visualize results
    print("Creating visualization...")
    visualize_sequence(
        single_pred[0],  # First batch
        target_sequence[0],  # First batch
        save_path='demo_visualization.png',
        show=True
    )
    
    # Save point clouds
    print("Saving point clouds...")
    save_point_cloud_sequence(
        single_pred[0],
        'demo_prediction.ply'
    )
    save_point_cloud_sequence(
        target_sequence[0],
        'demo_target.ply'
    )
    
    print("Demo files saved: demo_visualization.png, demo_prediction.ply, demo_target.ply")


def demo_uncertainty():
    """Demonstrate uncertainty estimation."""
    print("\n" + "="*50)
    print("DEMO: Uncertainty Estimation")
    print("="*50)
    
    # Create model with uncertainty
    model = S2Net(
        input_dim=3,
        hidden_dim=64,
        latent_dim=32,
        num_points=256,
        num_lstm_layers=1,
        num_pyramid_levels=2,
        use_temporal_variational=True,
        use_multi_scale=False,
        use_uncertainty=True,  # Enable uncertainty
        dropout=0.1
    )
    
    # Create demo data
    input_sequence, target_sequence = create_demo_data(batch_size=1, seq_len=5, num_points=256)
    
    # Generate predictions with uncertainty
    model.eval()
    with torch.no_grad():
        predictions = model(input_sequence, is_training=False, return_uncertainty=True)
    
    pred_points = predictions['predicted_point_clouds']
    uncertainties = predictions['uncertainties']
    
    print(f"Prediction shape: {pred_points.shape}")
    print(f"Uncertainty shape: {uncertainties.shape}")
    print(f"Mean uncertainty: {uncertainties.mean().item():.4f}")
    print(f"Std uncertainty: {uncertainties.std().item():.4f}")
    
    # Visualize uncertainty
    from utils.visualization import visualize_uncertainty
    visualize_uncertainty(
        pred_points[0],
        uncertainties[0],
        save_path='demo_uncertainty.png',
        show=True
    )


def main():
    """Main demo function."""
    print("S2Net: Stochastic Sequential Pointcloud Forecasting - Demo")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Demo 1: Model creation
        model, input_seq, target_seq, predictions = demo_model_creation()
        
        # Demo 2: Training
        trained_model = demo_training()
        
        # Demo 3: Prediction
        demo_prediction()
        
        # Demo 4: Uncertainty (if supported)
        try:
            demo_uncertainty()
        except Exception as e:
            print(f"Uncertainty demo failed: {e}")
        
        print("\n" + "="*70)
        print("Demo completed successfully!")
        print("Check the generated files:")
        print("- demo_visualization.png: Visualization of predictions")
        print("- demo_prediction.ply: Predicted point clouds")
        print("- demo_target.ply: Target point clouds")
        print("- demo_uncertainty.png: Uncertainty visualization (if available)")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
