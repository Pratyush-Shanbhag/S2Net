#!/usr/bin/env python3
"""
Training script for Two-Stream S2Net on KITTI dataset.
Uses sequences 1-5 for training and 6-7 for validation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2net_two_stream import TwoStreamS2Net
from utils.improved_losses import CombinedTwoStreamLoss
from data.kitti_dataloader import create_kitti_dataloaders
from utils.metrics import chamfer_distance, hausdorff_distance
# from utils.visualization import visualize_point_cloud_sequence


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Dict[str, Any], device: str) -> TwoStreamS2Net:
    """Create Two-Stream S2Net model."""
    model = TwoStreamS2Net(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        num_points=config['model']['num_points'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        num_pyramid_levels=config['model']['num_pyramid_levels'],
        use_temporal_variational=config['model']['use_temporal_variational'],
        use_multi_scale=config['model']['use_multi_scale'],
        dropout=config['model']['dropout']
    )
    
    model = model.to(device)
    print(f"Model created with {model.get_model_size():,} parameters")
    return model


def create_optimizer(model: TwoStreamS2Net, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer for the model."""
    optimizer_type = config['training']['optimizer']['type']
    lr = config['training']['optimizer']['lr']
    weight_decay = config['training']['optimizer']['weight_decay']
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = config['training']['optimizer'].get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    scheduler_type = config['training']['scheduler']['type']
    
    if scheduler_type.lower() == 'step':
        step_size = config['training']['scheduler']['step_size']
        gamma = config['training']['scheduler']['gamma']
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        T_max = config['training']['scheduler']['T_max']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type.lower() == 'plateau':
        patience = config['training']['scheduler']['patience']
        factor = config['training']['scheduler']['factor']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model: TwoStreamS2Net, 
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                loss_fn: CombinedTwoStreamLoss,
                device: str,
                epoch: int,
                config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_det_loss = 0.0
    total_stoch_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    
    num_batches = 0
    
    # KL annealing
    kl_weight = config['training']['loss']['kl_weight']
    if config['training']['loss']['use_annealing']:
        annealing_steps = config['training']['loss']['annealing_steps']
        kl_weight = min(kl_weight, epoch * kl_weight / annealing_steps)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        input_seq = batch['input_sequence'].to(device)  # [batch_size, seq_len, num_points, 3]
        target_seq = batch['target_sequence'].to(device)  # [batch_size, pred_len, num_points, 3]
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(input_seq, is_training=True)
        
        # Compute loss
        loss_dict = loss_fn(
            predictions=predictions,
            target_points=target_seq
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_det_loss += loss_dict['deterministic_loss'].item()
        total_stoch_loss += loss_dict['stochastic_loss'].item()
        total_kl_loss += loss_dict['kl_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.6f}",
            'Det': f"{loss_dict['deterministic_loss'].item():.6f}",
            'Stoch': f"{loss_dict['stochastic_loss'].item():.6f}",
            'KL': f"{loss_dict['kl_loss'].item():.6f}"
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'deterministic_loss': total_det_loss / num_batches,
        'stochastic_loss': total_stoch_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches
    }


def validate_epoch(model: TwoStreamS2Net,
                   dataloader: DataLoader,
                   loss_fn: CombinedTwoStreamLoss,
                   device: str,
                   config: Dict[str, Any]) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_det_loss = 0.0
    total_stoch_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    total_chamfer_det = 0.0
    total_chamfer_stoch = 0.0
    total_hausdorff_det = 0.0
    total_hausdorff_stoch = 0.0
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_seq = batch['input_sequence'].to(device)
            target_seq = batch['target_sequence'].to(device)
            
            # Forward pass
            predictions = model(input_seq, is_training=False)
            
            # Compute loss
            loss_dict = loss_fn(
                predictions=predictions,
                target_points=target_seq
            )
            
            # Compute metrics
            det_pred = predictions['deterministic_predictions']
            stoch_pred = predictions['stochastic_predictions']
            
            chamfer_det = chamfer_distance(det_pred, target_seq).mean()
            chamfer_stoch = chamfer_distance(stoch_pred, target_seq).mean()
            hausdorff_det = hausdorff_distance(det_pred, target_seq).mean()
            hausdorff_stoch = hausdorff_distance(stoch_pred, target_seq).mean()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_det_loss += loss_dict['deterministic_loss'].item()
            total_stoch_loss += loss_dict['stochastic_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_chamfer_det += chamfer_det.item()
            total_chamfer_stoch += chamfer_stoch.item()
            total_hausdorff_det += hausdorff_det.item()
            total_hausdorff_stoch += hausdorff_stoch.item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'deterministic_loss': total_det_loss / num_batches,
        'stochastic_loss': total_stoch_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'chamfer_distance_deterministic': total_chamfer_det / num_batches,
        'chamfer_distance_stochastic': total_chamfer_stoch / num_batches,
        'hausdorff_distance_deterministic': total_hausdorff_det / num_batches,
        'hausdorff_distance_stochastic': total_hausdorff_stoch / num_batches
    }


def save_checkpoint(model: TwoStreamS2Net,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   metrics: Dict[str, float],
                   filepath: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'latent_dim': model.latent_dim,
            'num_points': model.num_points,
            'use_temporal_variational': model.use_temporal_variational,
            'use_multi_scale': model.use_multi_scale
        }
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stream S2Net on KITTI dataset')
    parser.add_argument('--config', type=str, 
                       default='/home/pratyush/ISyE_Research/S2Net/configs/kitti_two_stream_config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(42)
    
    # Create model
    model = create_model(config, device)
    
    # Create loss function
    loss_fn = CombinedTwoStreamLoss()
    
    # Create dataloaders
    print("Creating KITTI dataloaders...")
    train_dataloader, val_dataloader = create_kitti_dataloaders(config)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_total_loss', float('inf'))
    
    # Create save directory
    save_dir = '/home/pratyush/ISyE_Research/S2Net/checkpoints/kitti'
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, config)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataloader, loss_fn, device, config)
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['total_loss'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['total_loss']:.6f}")
        print(f"  - Deterministic: {train_metrics['deterministic_loss']:.6f}")
        print(f"  - Stochastic: {train_metrics['stochastic_loss']:.6f}")
        print(f"  - KL: {train_metrics['kl_loss']:.6f}")
        print(f"  - Reconstruction: {train_metrics['reconstruction_loss']:.6f}")
        
        print(f"Val Loss: {val_metrics['total_loss']:.6f}")
        print(f"  - Deterministic: {val_metrics['deterministic_loss']:.6f}")
        print(f"  - Stochastic: {val_metrics['stochastic_loss']:.6f}")
        print(f"  - KL: {val_metrics['kl_loss']:.6f}")
        print(f"  - Reconstruction: {val_metrics['reconstruction_loss']:.6f}")
        print(f"  - Chamfer Det: {val_metrics['chamfer_distance_deterministic']:.6f}")
        print(f"  - Chamfer Stoch: {val_metrics['chamfer_distance_stochastic']:.6f}")
        print(f"  - Hausdorff Det: {val_metrics['hausdorff_distance_deterministic']:.6f}")
        print(f"  - Hausdorff Stoch: {val_metrics['hausdorff_distance_stochastic']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
            print(f"New best model saved: {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, config['training']['num_epochs']-1, val_metrics, final_model_path)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    main()
