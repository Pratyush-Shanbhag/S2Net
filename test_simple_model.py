#!/usr/bin/env python3
"""
Test simple model creation and forward pass
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import S2Net

def test_simple_model():
    """Test simple model creation and forward pass."""
    print("Testing simple model...")
    
    try:
        # Create simple model
        model = S2Net(
            input_dim=3,
            hidden_dim=64,
            latent_dim=32,
            num_points=256,
            num_lstm_layers=1,
            num_pyramid_levels=1,
            use_temporal_variational=False,
            use_multi_scale=False,
            use_uncertainty=False,
            use_temporal_decoder=False,
            dropout=0.1
        )
        
        print(f"✓ Model created successfully with {model.get_model_size():,} parameters")
        
        # Test forward pass
        batch_size, seq_len, num_points = 2, 5, 256
        input_sequence = torch.randn(batch_size, seq_len, num_points, 3)
        
        print(f"Input shape: {input_sequence.shape}")
        
        model.eval()
        with torch.no_grad():
            predictions = model(input_sequence, is_training=False)
        
        print(f"✓ Forward pass successful")
        print(f"Predicted shape: {predictions['predicted_point_clouds'].shape}")
        print(f"Latent shape: {predictions['latent_samples'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_simple_model()
    if success:
        print("\n🎉 Simple model test passed!")
    else:
        print("\n❌ Simple model test failed!")
        sys.exit(1)
