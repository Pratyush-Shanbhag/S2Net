#!/usr/bin/env python3
"""
Test script for S2Net model on KITTI sequences 8, 9, and 10.
This script loads the trained model and tests it on the specified sequences.
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2net import S2Net
from data.dataloader import create_dataloader
from utils.metrics import chamfer_distance, hausdorff_distance, compute_metrics
import time

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

def load_model(device, config_path='configs/simple_kitti.yaml', checkpoint_path='checkpoints/best_model.pth'):
    """Load the trained S2Net model."""
    print("Loading model configuration...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    print(f"Model configuration: {model_config}")
    
    # Create model
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
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def test_sequence(model, dataloader, sequence_name, device, num_samples=3):
    """Test the model on a specific sequence."""
    print(f"\n=== Testing Sequence {sequence_name} ===")
    
    all_metrics = {
        'chamfer_distance': [],
        'hausdorff_distance': [],
        'l2_distance': [],
        'mae': [],
        'rmse': []
    }
    
    inference_times = []
    
    # Test on multiple samples from the sequence
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        print(f"  Sample {i+1}/{num_samples}")
        
        input_sequence = batch['input_sequence'].to(device)
        target_sequence = batch['target_sequence'].to(device)
        
        print(f"    Input shape: {input_sequence.shape}")
        print(f"    Target shape: {target_sequence.shape}")
        
        # Make prediction
        start_time = time.time()
        with torch.no_grad():
            predictions = model(input_sequence)
            
            if isinstance(predictions, dict):
                pred_points = predictions['predicted_point_clouds']
            else:
                pred_points = predictions
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        print(f"    Prediction shape: {pred_points.shape}")
        print(f"    Inference time: {inference_time:.4f}s")
        
        # Convert to numpy for metrics computation
        input_np = input_sequence.squeeze(0).cpu().numpy()
        target_np = target_sequence.squeeze(0).cpu().numpy()
        pred_np = pred_points.squeeze(0).cpu().numpy()
        
        # Compute metrics for each frame
        for frame_idx in range(min(len(target_np), len(pred_np))):
            target_frame = target_np[frame_idx]
            pred_frame = pred_np[frame_idx]
            
            # Convert to tensors with proper dimensions for metrics
            target_tensor = torch.FloatTensor(target_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, num_points, 3]
            pred_tensor = torch.FloatTensor(pred_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, num_points, 3]
            
            # Chamfer Distance
            chamfer_dist = chamfer_distance(pred_tensor, target_tensor).item()
            all_metrics['chamfer_distance'].append(chamfer_dist)
            
            # Hausdorff Distance
            hausdorff_dist = hausdorff_distance(pred_tensor, target_tensor).item()
            all_metrics['hausdorff_distance'].append(hausdorff_dist)
            
            # L2 Distance
            l2_dist = torch.norm(pred_tensor - target_tensor, dim=-1).mean().item()
            all_metrics['l2_distance'].append(l2_dist)
            
            # MAE and RMSE
            mae = np.mean(np.abs(target_frame - pred_frame))
            rmse = np.sqrt(np.mean((target_frame - pred_frame) ** 2))
            all_metrics['mae'].append(mae)
            all_metrics['rmse'].append(rmse)
        
        # Create visualization for first sample
        if i == 0:
            print("    Creating visualization...")
            
            # Input sequence
            input_titles = [f'Input Frame {j+1}' for j in range(len(input_np))]
            visualize_point_clouds(input_np, input_titles, 
                                 f'seq_{sequence_name}_input_sequence.png')
            
            # Target vs Prediction
            target_titles = [f'Target Frame {j+1}' for j in range(len(target_np))]
            pred_titles = [f'Predicted Frame {j+1}' for j in range(len(pred_np))]
            
            # Show first few frames of prediction
            pred_subset = pred_np[:len(target_np)]  # Match target length
            pred_subset_titles = [f'Predicted Frame {j+1}' for j in range(len(pred_subset))]
            
            visualize_point_clouds(target_np, target_titles, 
                                 f'seq_{sequence_name}_target_sequence.png')
            visualize_point_clouds(pred_subset, pred_subset_titles, 
                                 f'seq_{sequence_name}_predicted_sequence.png')
    
    # Compute average metrics
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        avg_metrics[metric_name] = np.mean(values)
    
    avg_inference_time = np.mean(inference_times)
    
    print(f"\n📊 Results for Sequence {sequence_name}:")
    print(f"  Chamfer Distance: {avg_metrics['chamfer_distance']:.3f}")
    print(f"  Hausdorff Distance: {avg_metrics['hausdorff_distance']:.3f}")
    print(f"  L2 Distance: {avg_metrics['l2_distance']:.3f}")
    print(f"  MAE: {avg_metrics['mae']:.3f}")
    print(f"  RMSE: {avg_metrics['rmse']:.3f}")
    print(f"  Avg Inference Time: {avg_inference_time:.4f}s")
    
    return avg_metrics, avg_inference_time

def main():
    """Main test function for sequences 8-10."""
    print("🚀 S2Net Testing on KITTI Sequences 8-10")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(device)
    
    # Test sequences
    sequences = ['08', '09', '10']
    all_results = {}
    
    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Testing Sequence {seq}")
        print(f"{'='*60}")
        
        try:
            # Create data loader for this sequence
            dataloader = create_dataloader(
                dataset_name='kitti',
                data_path='/home/pratyush/ISyE_Research/datasets/unzipped/KITTI/dataset',
                batch_size=1,
                sequence_length=5,
                prediction_length=3,
                num_points=512,
                num_workers=0,
                shuffle=False,
                is_training=False,
                kitti_sequences=[seq],
                max_sequence_length=50  # Limit for testing
            )
            
            print(f"✅ Data loader created for sequence {seq}")
            print(f"   Dataset size: {len(dataloader.dataset)}")
            
            # Test the sequence
            metrics, inference_time = test_sequence(model, dataloader, seq, device, num_samples=3)
            all_results[seq] = {
                'metrics': metrics,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"❌ Error testing sequence {seq}: {e}")
            import traceback
            traceback.print_exc()
            all_results[seq] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY RESULTS")
    print(f"{'='*60}")
    
    for seq, results in all_results.items():
        if results is not None:
            print(f"\nSequence {seq}:")
            print(f"  Chamfer Distance: {results['metrics']['chamfer_distance']:.3f}")
            print(f"  Hausdorff Distance: {results['metrics']['hausdorff_distance']:.3f}")
            print(f"  L2 Distance: {results['metrics']['l2_distance']:.3f}")
            print(f"  MAE: {results['metrics']['mae']:.3f}")
            print(f"  RMSE: {results['metrics']['rmse']:.3f}")
            print(f"  Avg Inference Time: {results['inference_time']:.4f}s")
        else:
            print(f"\nSequence {seq}: ❌ FAILED")
    
    # Compute overall averages
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        print(f"\n{'='*60}")
        print("📈 OVERALL AVERAGES")
        print(f"{'='*60}")
        
        avg_chamfer = np.mean([r['metrics']['chamfer_distance'] for r in valid_results.values()])
        avg_hausdorff = np.mean([r['metrics']['hausdorff_distance'] for r in valid_results.values()])
        avg_l2 = np.mean([r['metrics']['l2_distance'] for r in valid_results.values()])
        avg_mae = np.mean([r['metrics']['mae'] for r in valid_results.values()])
        avg_rmse = np.mean([r['metrics']['rmse'] for r in valid_results.values()])
        avg_inference = np.mean([r['inference_time'] for r in valid_results.values()])
        
        print(f"Average Chamfer Distance: {avg_chamfer:.3f}")
        print(f"Average Hausdorff Distance: {avg_hausdorff:.3f}")
        print(f"Average L2 Distance: {avg_l2:.3f}")
        print(f"Average MAE: {avg_mae:.3f}")
        print(f"Average RMSE: {avg_rmse:.3f}")
        print(f"Average Inference Time: {avg_inference:.4f}s")
    
    print(f"\n🎉 Testing completed!")
    print(f"Generated visualizations:")
    for seq in sequences:
        print(f"  - seq_{seq}_input_sequence.png")
        print(f"  - seq_{seq}_target_sequence.png")
        print(f"  - seq_{seq}_predicted_sequence.png")

if __name__ == "__main__":
    main()
