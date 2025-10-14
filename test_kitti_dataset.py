#!/usr/bin/env python3
"""
Test script to verify KITTI dataset loading
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloader

def test_kitti_dataset():
    """Test KITTI dataset loading."""
    print("Testing KITTI dataset loading...")
    
    try:
        # Create data loader for KITTI
        dataloader = create_dataloader(
            dataset_name='kitti',
            data_path='/home/pratyush/ISyE_Research/datasets/unzipped/KITTI/dataset',
            batch_size=2,
            sequence_length=10,
            prediction_length=5,
            num_points=1024,
            num_workers=0,  # Use 0 for testing
            shuffle=False,
            is_training=True,
            kitti_sequences=['00', '01', '02', '03', '04', '05'],
            max_sequence_length=100  # Limit for testing
        )
        
        print(f"✓ Data loader created successfully")
        print(f"  Dataset size: {len(dataloader.dataset)}")
        
        # Test loading a batch
        batch = next(iter(dataloader))
        
        input_seq = batch['input_sequence']
        target_seq = batch['target_sequence']
        
        print(f"✓ Batch loaded successfully")
        print(f"  Input shape: {input_seq.shape}")
        print(f"  Target shape: {target_seq.shape}")
        print(f"  Input range: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
        print(f"  Target range: [{target_seq.min():.3f}, {target_seq.max():.3f}]")
        
        # Test a few more batches
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test only first 3 batches
                break
            print(f"  Batch {i+1}: Input {batch['input_sequence'].shape}, Target {batch['target_sequence'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ KITTI dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_kitti_dataset()
    if success:
        print("\n🎉 KITTI dataset test passed!")
    else:
        print("\n❌ KITTI dataset test failed!")
        sys.exit(1)
