import torch
import argparse
import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from models import S2Net
from data import create_dataloader
from utils.metrics import evaluate_model, compute_sequence_metrics
from utils.visualization import visualize_sequence, save_point_cloud_sequence, visualize_uncertainty


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate S2Net model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
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
    
    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predicted point clouds')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/',
                       help='Directory to save results')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes')
    
    return parser.parse_args()


def load_model(model_path: str, device: str) -> S2Net:
    """Load trained model."""
    if os.path.exists(model_path):
        model = S2Net.load_model(model_path, device)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return model


def evaluate_single_sequence(model: S2Net,
                           input_sequence: torch.Tensor,
                           target_sequence: torch.Tensor,
                           device: str,
                           num_samples: int = 10) -> Dict[str, Any]:
    """
    Evaluate model on a single sequence.
    
    Args:
        model: Trained model
        input_sequence: Input point cloud sequence [1, seq_len, num_points, 3]
        target_sequence: Target point cloud sequence [1, seq_len, num_points, 3]
        device: Device to use
        num_samples: Number of samples to generate
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)
        
        # Generate single prediction
        predictions = model(input_sequence, is_training=False)
        single_pred = predictions['predicted_point_clouds']
        
        # Generate multiple samples
        if num_samples > 1:
            multi_pred = model.sample_future(
                input_sequence,
                future_steps=target_sequence.shape[1],
                num_samples=num_samples
            )
        else:
            multi_pred = single_pred.unsqueeze(1)  # [1, 1, seq_len, num_points, 3]
        
        # Compute metrics
        single_metrics = compute_sequence_metrics(single_pred, target_sequence)
        multi_metrics = compute_sequence_metrics(multi_pred, target_sequence)
        
        # Add uncertainty if available
        uncertainty_info = {}
        if 'uncertainties' in predictions:
            uncertainty_info['uncertainty_mean'] = predictions['uncertainties'].mean().item()
            uncertainty_info['uncertainty_std'] = predictions['uncertainties'].std().item()
        
        return {
            'single_prediction': single_pred,
            'multi_predictions': multi_pred,
            'target': target_sequence,
            'single_metrics': single_metrics,
            'multi_metrics': multi_metrics,
            'uncertainty_info': uncertainty_info
        }


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config.update(vars(args))
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create data loader
    dataloader = create_dataloader(
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
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    # Evaluate model
    all_metrics = []
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_sequence = batch['input_sequence']
        target_sequence = batch['target_sequence']
        
        # Evaluate each sequence in the batch
        for i in range(input_sequence.shape[0]):
            single_input = input_sequence[i:i+1]  # [1, seq_len, num_points, 3]
            single_target = target_sequence[i:i+1]  # [1, seq_len, num_points, 3]
            
            # Evaluate single sequence
            results = evaluate_single_sequence(
                model, single_input, single_target, device, args.num_samples
            )
            
            all_results.append(results)
            all_metrics.append(results['multi_metrics'])
            
            # Save predictions if requested
            if args.save_predictions:
                pred_dir = os.path.join(args.output_dir, f'batch_{batch_idx}_seq_{i}')
                os.makedirs(pred_dir, exist_ok=True)
                
                # Save single prediction
                save_point_cloud_sequence(
                    results['single_prediction'][0],
                    os.path.join(pred_dir, 'single_prediction.ply')
                )
                
                # Save multiple predictions
                for s in range(min(args.num_samples, results['multi_predictions'].shape[1])):
                    save_point_cloud_sequence(
                        results['multi_predictions'][0, s],
                        os.path.join(pred_dir, f'multi_prediction_{s}.ply')
                    )
                
                # Save target
                save_point_cloud_sequence(
                    results['target'][0],
                    os.path.join(pred_dir, 'target.ply')
                )
            
            # Save visualizations if requested
            if args.save_visualizations and batch_idx < 5:  # Only save first 5 batches
                vis_dir = os.path.join(args.output_dir, f'visualizations_batch_{batch_idx}_seq_{i}')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Visualize single prediction
                visualize_sequence(
                    results['single_prediction'][0],
                    results['target'][0],
                    save_path=os.path.join(vis_dir, 'single_prediction.png'),
                    show=False
                )
                
                # Visualize uncertainty if available
                if 'uncertainties' in results and results['uncertainties'] is not None:
                    visualize_uncertainty(
                        results['single_prediction'][0],
                        results['uncertainties'][0],
                        save_path=os.path.join(vis_dir, 'uncertainty.png'),
                        show=False
                    )
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("S2Net Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Number of sequences: {len(all_results)}\n")
        f.write("\nMetrics:\n")
        for metric_name, value in avg_metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Print uncertainty statistics if available
    uncertainty_means = []
    uncertainty_stds = []
    for result in all_results:
        if result['uncertainty_info']:
            uncertainty_means.append(result['uncertainty_info']['uncertainty_mean'])
            uncertainty_stds.append(result['uncertainty_info']['uncertainty_std'])
    
    if uncertainty_means:
        print(f"\nUncertainty Statistics:")
        print(f"Mean uncertainty: {np.mean(uncertainty_means):.6f}")
        print(f"Std uncertainty: {np.mean(uncertainty_stds):.6f}")


if __name__ == '__main__':
    main()
