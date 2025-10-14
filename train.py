import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import yaml
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

from models import S2Net
from data import create_dataloader, get_dataset_info
from utils.losses import ChamferDistanceLoss, KLDivergenceLoss, CombinedLoss
from utils.metrics import compute_metrics
from utils.visualization import save_point_cloud_sequence


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train S2Net model')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model_name', type=str, default='s2net',
                       help='Name of the model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['kitti', 'nuscenes', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Length of input sequence')
    parser.add_argument('--prediction_length', type=int, default=5,
                       help='Length of prediction sequence')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points per point cloud')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                       help='Weight for KL divergence loss')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--validate_every', type=int, default=10,
                       help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=20,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"Config file {config_path} not found, using default config")
        return {}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Dict[str, Any], device: str) -> S2Net:
    """Create S2Net model."""
    model_config = config.get('model', {})
    
    model = S2Net(
        input_dim=model_config.get('input_dim', 3),
        hidden_dim=model_config.get('hidden_dim', 256),
        latent_dim=model_config.get('latent_dim', 128),
        num_points=model_config.get('num_points', 1024),
        num_lstm_layers=model_config.get('num_lstm_layers', 2),
        num_pyramid_levels=model_config.get('num_pyramid_levels', 3),
        use_temporal_variational=model_config.get('use_temporal_variational', True),
        use_multi_scale=model_config.get('use_multi_scale', True),
        use_uncertainty=model_config.get('use_uncertainty', True),
        use_temporal_decoder=model_config.get('use_temporal_decoder', True),
        dropout=model_config.get('dropout', 0.1)
    )
    
    model = model.to(device)
    return model


def create_optimizer(model: S2Net, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer."""
    optimizer_config = config.get('optimizer', {})
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config.get('lr', 1e-4),
        weight_decay=optimizer_config.get('weight_decay', 1e-5),
        betas=optimizer_config.get('betas', (0.9, 0.999))
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    scheduler_config = config.get('scheduler', {})
    
    if scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_config.get('type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10)
        )
    
    return scheduler


def train_epoch(model: S2Net, 
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: str,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
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
        loss_dict = model.compute_loss(
            predictions,
            target_sequence,
            kl_weight=criterion.kl_weight,
            reconstruction_weight=criterion.reconstruction_weight
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
        total_kl_loss += loss_dict['kl_loss'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
            'KL': f'{loss_dict["kl_loss"].item():.4f}'
        })
    
    # Compute average losses
    avg_loss = total_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'kl_loss': avg_kl_loss
    }


def validate_epoch(model: S2Net,
                  dataloader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: str,
                  epoch: int) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            
            # Forward pass
            predictions = model(input_sequence, is_training=False)
            
            # Compute loss
            loss_dict = model.compute_loss(
                predictions,
                target_sequence,
                kl_weight=criterion.kl_weight,
                reconstruction_weight=criterion.reconstruction_weight
            )
            
            loss = loss_dict['total_loss']
            
            # Compute metrics
            metrics = compute_metrics(
                predictions['predicted_point_clouds'],
                target_sequence
            )
            
            # Update totals
            total_loss += loss.item()
            total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{loss_dict["reconstruction_loss"].item():.4f}',
                'KL': f'{loss_dict["kl_loss"].item():.4f}'
            })
    
    # Compute average metrics
    avg_loss = total_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'kl_loss': avg_kl_loss,
        **avg_metrics
    }


def save_checkpoint(model: S2Net,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   save_dir: str,
                   is_best: bool = False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved at epoch {epoch}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.update(vars(args))
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config, device)
    print(f"Model created with {model.get_model_size():,} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = CombinedLoss(
        kl_weight=args.kl_weight,
        reconstruction_weight=args.reconstruction_weight
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        num_points=args.num_points,
        num_workers=args.num_workers,
        shuffle=True,
        is_training=True
    )
    
    val_loader = create_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        num_points=args.num_points,
        num_workers=args.num_workers,
        shuffle=False,
        is_training=False
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs-1}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Validate
        if epoch % args.validate_every == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Update learning rate
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_loss
            if is_best:
                best_loss = val_metrics['loss']
            
            if epoch % args.save_every == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics['loss'],
                    args.save_dir, is_best
                )
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        if epoch % args.validate_every == 0:
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Reconstruction Loss: {val_metrics['reconstruction_loss']:.4f}")
            print(f"Val KL Loss: {val_metrics['kl_loss']:.4f}")
    
    # Save final model
    model.save_model(os.path.join(args.save_dir, 'final_model.pth'))
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
