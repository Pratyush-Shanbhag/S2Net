import torch
import torch.utils.data as data
from typing import Dict, Any, Optional, Tuple
from .dataset import PointCloudDataset, KITTIDataset, NuScenesDataset, SyntheticDataset, PointCloudSequenceDataset
from .transforms import PointCloudTransforms, get_default_transforms


def create_dataloader(dataset_name: str,
                     data_path: str,
                     batch_size: int = 8,
                     sequence_length: int = 10,
                     prediction_length: int = 5,
                     num_points: int = 1024,
                     num_workers: int = 4,
                     shuffle: bool = True,
                     is_training: bool = True,
                     transforms: Optional[PointCloudTransforms] = None,
                     **kwargs) -> data.DataLoader:
    """
    Create a DataLoader for point cloud sequences.
    
    Args:
        dataset_name: Name of the dataset ('kitti', 'nuscenes', 'synthetic')
        data_path: Path to the dataset
        batch_size: Batch size
        sequence_length: Length of input sequence
        prediction_length: Length of prediction sequence
        num_points: Number of points per point cloud
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        is_training: Whether in training mode
        transforms: Custom transforms to apply
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader object
    """
    # Get default transforms if not provided
    if transforms is None:
        transforms = get_default_transforms(is_training)
    
    # Create dataset based on name
    if dataset_name.lower() == 'kitti':
        dataset = KITTIDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms,
            split='train' if is_training else 'val',
            kitti_sequences=kwargs.get('kitti_sequences', ['00', '01', '02', '03', '04', '05']),
            max_sequence_length=kwargs.get('max_sequence_length', None),
            **{k: v for k, v in kwargs.items() if k not in ['kitti_sequences', 'max_sequence_length']}
        )
    elif dataset_name.lower() == 'nuscenes':
        dataset = NuScenesDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms,
            split='train' if is_training else 'val',
            **kwargs
        )
    elif dataset_name.lower() == 'synthetic':
        dataset = SyntheticDataset(
            num_sequences=kwargs.get('num_sequences', 1000),
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create DataLoader
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_sequence_dataloader(dataset_name: str,
                              data_path: str,
                              batch_size: int = 4,
                              sequence_length: int = 10,
                              prediction_length: int = 5,
                              num_points: int = 1024,
                              num_workers: int = 4,
                              shuffle: bool = True,
                              is_training: bool = True,
                              sequence_stride: int = 1,
                              transforms: Optional[PointCloudTransforms] = None,
                              **kwargs) -> data.DataLoader:
    """
    Create a DataLoader for complete sequences.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset
        batch_size: Batch size
        sequence_length: Length of input sequence
        prediction_length: Length of prediction sequence
        num_points: Number of points per point cloud
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        is_training: Whether in training mode
        sequence_stride: Stride for sequence sampling
        transforms: Custom transforms to apply
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader object
    """
    # Get default transforms if not provided
    if transforms is None:
        transforms = get_default_transforms(is_training)
    
    # Create base dataset
    if dataset_name.lower() == 'kitti':
        base_dataset = KITTIDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms,
            split='train' if is_training else 'val',
            **kwargs
        )
    elif dataset_name.lower() == 'nuscenes':
        base_dataset = NuScenesDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms,
            split='train' if is_training else 'val',
            **kwargs
        )
    elif dataset_name.lower() == 'synthetic':
        base_dataset = SyntheticDataset(
            num_sequences=kwargs.get('num_sequences', 1000),
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            num_points=num_points,
            transforms=transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create sequence dataset
    dataset = PointCloudSequenceDataset(
        base_dataset=base_dataset,
        sequence_stride=sequence_stride
    )
    
    # Create DataLoader
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching point cloud sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Stack input and target sequences
    input_sequences = torch.stack([item['input_sequence'] for item in batch])
    target_sequences = torch.stack([item['target_sequence'] for item in batch])
    
    return {
        'input_sequence': input_sequences,
        'target_sequence': target_sequences
    }


def create_dataloaders(dataset_name: str,
                      data_path: str,
                      batch_size: int = 8,
                      sequence_length: int = 10,
                      prediction_length: int = 5,
                      num_points: int = 1024,
                      num_workers: int = 4,
                      train_split: float = 0.8,
                      val_split: float = 0.1,
                      test_split: float = 0.1,
                      **kwargs) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset
        batch_size: Batch size
        sequence_length: Length of input sequence
        prediction_length: Length of prediction sequence
        num_points: Number of points per point cloud
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create training dataloader
    train_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        num_points=num_points,
        num_workers=num_workers,
        shuffle=True,
        is_training=True,
        **kwargs
    )
    
    # Create validation dataloader
    val_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        num_points=num_points,
        num_workers=num_workers,
        shuffle=False,
        is_training=False,
        **kwargs
    )
    
    # Create test dataloader
    test_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        num_points=num_points,
        num_workers=num_workers,
        shuffle=False,
        is_training=False,
        **kwargs
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_info = {
        'kitti': {
            'name': 'KITTI',
            'description': 'KITTI autonomous driving dataset',
            'default_sequence_length': 10,
            'default_prediction_length': 5,
            'default_num_points': 1024,
            'file_format': '.bin',
            'coordinate_system': 'camera'
        },
        'nuscenes': {
            'name': 'nuScenes',
            'description': 'nuScenes autonomous driving dataset',
            'default_sequence_length': 10,
            'default_prediction_length': 5,
            'default_num_points': 1024,
            'file_format': '.pcd.bin',
            'coordinate_system': 'lidar'
        },
        'synthetic': {
            'name': 'Synthetic',
            'description': 'Synthetic point cloud dataset for testing',
            'default_sequence_length': 10,
            'default_prediction_length': 5,
            'default_num_points': 1024,
            'file_format': 'generated',
            'coordinate_system': 'world'
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {})
