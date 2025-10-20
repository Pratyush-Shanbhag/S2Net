#!/usr/bin/env python3
"""
Training script for Two-Stream S2Net architecture.
Implements proper two-stream training as described in the paper.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.s2net_two_stream import TwoStreamS2Net
from utils.improved_losses import CombinedTwoStreamLoss
from data.dataloader import create_sequence_dataloader
from utils.visualization import visualize_point_cloud_sequence
from utils.metrics import chamfer_distance, hausdorff_distance


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
                dataloader: data.DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: str,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_deterministic_loss = 0.0
    total_stochastic_loss = 0.0
    total_kl_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        input_sequence = batch['input_sequence'].to(device)
        target_sequence = batch['target_sequence'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        predictions = model(input_sequence, is_training=True)
        
        # Compute loss
        loss_dict = criterion(predictions, target_sequence)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
        total_deterministic_loss += loss_dict['deterministic_loss'].item()
        total_stochastic_loss += loss_dict['stochastic_loss'].item()
        total_kl_loss += loss_dict['kl_loss'].item()
        total_temporal_loss += loss_dict['temporal_loss'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
            'Det': f'{loss_dict["deterministic_loss"].item():.4f}',
            'Stoch': f'{loss_dict["stochastic_loss"].item():.4f}',
            'KL': f'{loss_dict["kl_loss"].item():.4f}'
        })
    
    # Compute average losses
    avg_loss = total_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_deterministic_loss = total_deterministic_loss / num_batches
    avg_stochastic_loss = total_stochastic_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_temporal_loss = total_temporal_loss / num_batches
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'deterministic_loss': avg_deterministic_loss,
        'stochastic_loss': avg_stochastic_loss,
        'kl_loss': avg_kl_loss,
        'temporal_loss': avg_temporal_loss
    }


def validate_epoch(model: TwoStreamS2Net,
                  dataloader: data.DataLoader,
                  criterion: nn.Module,
                  device: str,
                  epoch: int) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_deterministic_loss = 0.0
    total_stochastic_loss = 0.0
    total_kl_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0
    
    # Metrics for evaluation
    total_chamfer_det = 0.0
    total_chamfer_stoch = 0.0
    total_hausdorff_det = 0.0
    total_hausdorff_stoch = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            
            # Forward pass
            predictions = model(input_sequence, is_training=False)
            
            # Compute loss
            loss_dict = criterion(predictions, target_sequence)
            loss = loss_dict['total_loss']
            
            # Update metrics
            total_loss += loss.item()
            total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            total_deterministic_loss += loss_dict['deterministic_loss'].item()
            total_stochastic_loss += loss_dict['stochastic_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            total_temporal_loss += loss_dict['temporal_loss'].item()
            num_batches += 1
            
            # Compute evaluation metrics
            det_pred = predictions['deterministic_predictions']
            stoch_pred = predictions['stochastic_predictions']
            
            # Chamfer distances
            chamfer_det = chamfer_distance(det_pred, target_sequence).mean().item()
            chamfer_stoch = chamfer_distance(stoch_pred, target_sequence).mean().item()
            
            # Hausdorff distances
            hausdorff_det = hausdorff_distance(det_pred, target_sequence).mean().item()
            hausdorff_stoch = hausdorff_distance(stoch_pred, target_sequence).mean().item()
            
            total_chamfer_det += chamfer_det
            total_chamfer_stoch += chamfer_stoch
            total_hausdorff_det += hausdorff_det
            total_hausdorff_stoch += hausdorff_stoch
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CD_Det': f'{chamfer_det:.4f}',
                'CD_Stoch': f'{chamfer_stoch:.4f}'
            })
    
    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_deterministic_loss = total_deterministic_loss / num_batches
    avg_stochastic_loss = total_stochastic_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_temporal_loss = total_temporal_loss / num_batches
    
    avg_chamfer_det = total_chamfer_det / num_batches
    avg_chamfer_stoch = total_chamfer_stoch / num_batches
    avg_hausdorff_det = total_hausdorff_det / num_batches
    avg_hausdorff_stoch = total_hausdorff_stoch / num_batches
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'deterministic_loss': avg_deterministic_loss,
        'stochastic_loss': avg_stochastic_loss,
        'kl_loss': avg_kl_loss,
        'temporal_loss': avg_temporal_loss,
        'chamfer_det': avg_chamfer_det,
        'chamfer_stoch': avg_chamfer_stoch,
        'hausdorff_det': avg_hausdorff_det,
        'hausdorff_stoch': avg_hausdorff_stoch
    }


def save_checkpoint(model: TwoStreamS2Net, 
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, 
                   model: TwoStreamS2Net,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler = None) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stream S2Net')
    parser.add_argument('--config', type=str, default='configs/two_stream_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = CombinedTwoStreamLoss(
        kl_weight=config['training']['loss']['kl_weight'],
        reconstruction_weight=config['training']['loss']['reconstruction_weight'],
        deterministic_weight=config['training']['loss']['deterministic_weight'],
        stochastic_weight=config['training']['loss']['stochastic_weight'],
        temporal_weight=config['training']['loss']['temporal_weight'],
        use_annealing=config['training']['loss']['use_annealing'],
        annealing_steps=config['training']['loss']['annealing_steps']
    )
    
    # Create data loaders
    train_loader = create_sequence_dataloader(
        dataset_name=config['data']['dataset_name'],
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        prediction_length=config['data']['prediction_length'],
        num_points=config['data']['num_points'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
        is_training=True
    )
    
    val_loader = create_sequence_dataloader(
        dataset_name=config['data']['dataset_name'],
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        prediction_length=config['data']['prediction_length'],
        num_points=config['data']['num_points'],
        num_workers=config['data']['num_workers'],
        shuffle=False,
        is_training=False
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_metrics)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        val_losses.append(val_metrics)
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"  - Reconstruction: {train_metrics['reconstruction_loss']:.4f}")
        print(f"  - Deterministic: {train_metrics['deterministic_loss']:.4f}")
        print(f"  - Stochastic: {train_metrics['stochastic_loss']:.4f}")
        print(f"  - KL: {train_metrics['kl_loss']:.4f}")
        print(f"  - Temporal: {train_metrics['temporal_loss']:.4f}")
        
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  - Chamfer Det: {val_metrics['chamfer_det']:.4f}")
        print(f"  - Chamfer Stoch: {val_metrics['chamfer_stoch']:.4f}")
        print(f"  - Hausdorff Det: {val_metrics['hausdorff_det']:.4f}")
        print(f"  - Hausdorff Stoch: {val_metrics['hausdorff_stoch']:.4f}")
        
        # Save checkpoint
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'],
                          f"checkpoints/two_stream_best_model.pth")
            print("New best model saved!")
        
        # Save regular checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'],
                          f"checkpoints/two_stream_epoch_{epoch}.pth")
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'],
                  "checkpoints/two_stream_final_model.pth")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
