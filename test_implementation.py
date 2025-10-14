#!/usr/bin/env python3
"""
Test script to verify S2Net implementation
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import S2Net
from data import create_dataloader
from utils.losses import CombinedLoss
from utils.metrics import compute_metrics


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    try:
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
        
        print(f"✓ Model created successfully with {model.get_model_size():,} parameters")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass."""
    print("Testing forward pass...")
    
    try:
        # Create dummy data
        batch_size, seq_len, num_points = 2, 5, 256
        input_sequence = torch.randn(batch_size, seq_len, num_points, 3)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(input_sequence, is_training=False)
        
        expected_keys = ['predicted_point_clouds', 'latent_samples', 'prior_mean', 
                        'prior_log_var', 'posterior_mean', 'posterior_log_var']
        
        for key in expected_keys:
            if key not in predictions:
                print(f"✗ Missing key in predictions: {key}")
                return False
        
        print(f"✓ Forward pass successful")
        print(f"  Predicted shape: {predictions['predicted_point_clouds'].shape}")
        print(f"  Latent shape: {predictions['latent_samples'].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_training_mode(model):
    """Test training mode."""
    print("Testing training mode...")
    
    try:
        # Create dummy data
        batch_size, seq_len, num_points = 2, 5, 256
        input_sequence = torch.randn(batch_size, seq_len, num_points, 3)
        target_sequence = torch.randn(batch_size, seq_len, num_points, 3)
        
        # Training forward pass
        model.train()
        predictions = model(input_sequence, is_training=True)
        
        # Compute loss
        loss_dict = model.compute_loss(predictions, target_sequence)
        
        expected_loss_keys = ['total_loss', 'reconstruction_loss', 'kl_loss']
        for key in expected_loss_keys:
            if key not in loss_dict:
                print(f"✗ Missing key in loss: {key}")
                return False
        
        print(f"✓ Training mode successful")
        print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"  Reconstruction loss: {loss_dict['reconstruction_loss'].item():.4f}")
        print(f"  KL loss: {loss_dict['kl_loss'].item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Training mode failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions."""
    print("Testing loss functions...")
    
    try:
        from utils.losses import ChamferDistanceLoss, KLDivergenceLoss, CombinedLoss
        
        # Test Chamfer Distance
        chamfer_loss = ChamferDistanceLoss()
        pred_points = torch.randn(2, 5, 256, 3)
        target_points = torch.randn(2, 5, 256, 3)
        chamfer_value = chamfer_loss(pred_points, target_points)
        print(f"✓ Chamfer Distance: {chamfer_value.item():.4f}")
        
        # Test KL Divergence
        kl_loss = KLDivergenceLoss()
        prior_mean = torch.randn(2, 5, 32)
        prior_log_var = torch.randn(2, 5, 32)
        posterior_mean = torch.randn(2, 5, 32)
        posterior_log_var = torch.randn(2, 5, 32)
        kl_value = kl_loss(prior_mean, prior_log_var, posterior_mean, posterior_log_var)
        print(f"✓ KL Divergence: {kl_value.item():.4f}")
        
        # Test Combined Loss
        combined_loss = CombinedLoss()
        predictions = {
            'predicted_point_clouds': pred_points,
            'prior_mean': prior_mean,
            'prior_log_var': prior_log_var,
            'posterior_mean': posterior_mean,
            'posterior_log_var': posterior_log_var
        }
        combined_value = combined_loss(predictions, target_points)
        print(f"✓ Combined Loss: {combined_value['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("Testing metrics...")
    
    try:
        pred_points = torch.randn(2, 5, 256, 3)
        target_points = torch.randn(2, 5, 256, 3)
        
        metrics = compute_metrics(pred_points, target_points)
        
        expected_metrics = ['chamfer_distance', 'hausdorff_distance', 'temporal_consistency',
                           'pred_density', 'target_density', 'density_ratio', 'l2_distance', 'mae', 'rmse']
        
        for metric in expected_metrics:
            if metric not in metrics:
                print(f"✗ Missing metric: {metric}")
                return False
        
        print("✓ Metrics computed successfully")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics failed: {e}")
        return False


def test_data_loader():
    """Test data loader."""
    print("Testing data loader...")
    
    try:
        # Test synthetic dataset
        dataloader = create_dataloader(
            dataset_name='synthetic',
            data_path='',
            batch_size=2,
            sequence_length=5,
            prediction_length=3,
            num_points=256,
            num_workers=0,  # Use 0 for testing
            shuffle=False,
            is_training=False,
            num_sequences=10  # Small number for testing
        )
        
        # Test loading a batch
        batch = next(iter(dataloader))
        
        if 'input_sequence' not in batch or 'target_sequence' not in batch:
            print("✗ Missing keys in batch")
            return False
        
        input_seq = batch['input_sequence']
        target_seq = batch['target_sequence']
        
        if input_seq.shape[0] != 2 or target_seq.shape[0] != 2:
            print("✗ Incorrect batch size")
            return False
        
        print(f"✓ Data loader successful")
        print(f"  Input shape: {input_seq.shape}")
        print(f"  Target shape: {target_seq.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        return False


def test_cuda():
    """Test CUDA functionality."""
    print("Testing CUDA...")
    
    if not torch.cuda.is_available():
        print("✓ CUDA not available, skipping CUDA tests")
        return True
    
    try:
        # Test model on CUDA
        model = S2Net(
            input_dim=3,
            hidden_dim=32,
            latent_dim=16,
            num_points=128,
            num_lstm_layers=1,
            num_pyramid_levels=2,
            use_temporal_variational=True,
            use_multi_scale=False,
            use_uncertainty=False,
            dropout=0.1
        )
        
        model = model.cuda()
        
        # Test forward pass on CUDA
        input_sequence = torch.randn(1, 3, 128, 3).cuda()
        
        with torch.no_grad():
            predictions = model(input_sequence, is_training=False)
        
        print("✓ CUDA functionality successful")
        return True
        
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def main():
    """Main test function."""
    print("S2Net Implementation Test")
    print("="*40)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
        ("Data Loader", test_data_loader),
        ("CUDA", test_cuda)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        try:
            if test_name == "Model Creation":
                model = test_func()
                results.append((test_name, model is not None))
                if model is not None:
                    # Test forward pass and training with the created model
                    results.append(("Forward Pass", test_forward_pass(model)))
                    results.append(("Training Mode", test_training_mode(model)))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
