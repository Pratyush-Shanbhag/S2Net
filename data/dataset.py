import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
from typing import List, Tuple, Optional, Dict, Any
import random


class PointCloudDataset(data.Dataset):
    """
    Base dataset class for point cloud sequences.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 10,
                 prediction_length: int = 5,
                 num_points: int = 1024,
                 transforms: Optional[Any] = None):
        super(PointCloudDataset, self).__init__()
        
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_points = num_points
        self.transforms = transforms
        
        # Load data file paths
        self.data_files = self._load_data_files()
        
    def _load_data_files(self) -> List[str]:
        """Load list of data files. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from file. To be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of point clouds.
        
        Returns:
            Dictionary containing:
                - input_sequence: Input point cloud sequence
                - target_sequence: Target point cloud sequence
        """
        # Load sequence of point clouds
        total_length = self.sequence_length + self.prediction_length
        sequence_files = self.data_files[idx:idx + total_length]
        
        if len(sequence_files) < total_length:
            # Pad with the last available file if sequence is too short
            sequence_files.extend([sequence_files[-1]] * (total_length - len(sequence_files)))
        
        point_clouds = []
        for file_path in sequence_files:
            pc = self._load_point_cloud(file_path)
            point_clouds.append(pc)
        
        # Convert to numpy array
        point_clouds = np.stack(point_clouds, axis=0)  # [total_length, num_points, 3]
        
        # Split into input and target sequences
        input_sequence = point_clouds[:self.sequence_length]
        target_sequence = point_clouds[self.sequence_length:]
        
        # Apply transforms if provided
        if self.transforms is not None:
            input_sequence = self.transforms(input_sequence)
            target_sequence = self.transforms(target_sequence)
        
        # Convert to tensors
        input_sequence = torch.from_numpy(input_sequence).float()
        target_sequence = torch.from_numpy(target_sequence).float()
        
        return {
            'input_sequence': input_sequence,
            'target_sequence': target_sequence
        }


class KITTIDataset(PointCloudDataset):
    """
    Dataset for KITTI point cloud sequences.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 10,
                 prediction_length: int = 5,
                 num_points: int = 1024,
                 transforms: Optional[Any] = None,
                 split: str = 'train',
                 kitti_sequences: Optional[List[str]] = None,
                 max_sequence_length: Optional[int] = None):
        self.split = split
        self.kitti_sequences = kitti_sequences or ['00', '01', '02', '03', '04', '05']
        self.max_sequence_length = max_sequence_length
        super(KITTIDataset, self).__init__(
            data_path, sequence_length, prediction_length, num_points, transforms
        )
        
    def _load_data_files(self) -> List[str]:
        """Load KITTI data files."""
        data_files = []
        
        # KITTI structure: data_path/sequences/XX/velodyne/*.bin
        sequences_dir = os.path.join(self.data_path, 'sequences')
        
        if not os.path.exists(sequences_dir):
            raise ValueError(f"KITTI sequences directory not found: {sequences_dir}")
        
        # Use only specified sequences
        for seq_dir in self.kitti_sequences:
            velodyne_dir = os.path.join(sequences_dir, seq_dir, 'velodyne')
            if os.path.exists(velodyne_dir):
                # Get all .bin files in this sequence
                bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
                
                # Limit sequence length if specified
                if self.max_sequence_length is not None:
                    bin_files = bin_files[:self.max_sequence_length]
                
                for bin_file in bin_files:
                    data_files.append(os.path.join(velodyne_dir, bin_file))
            else:
                print(f"Warning: Sequence {seq_dir} not found in {velodyne_dir}")
        
        print(f"Loaded {len(data_files)} point cloud files from sequences: {self.kitti_sequences}")
        return data_files
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load KITTI point cloud from .bin file."""
        # KITTI .bin files contain float32 data with x, y, z, intensity
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
        # Take only x, y, z coordinates (ignore intensity)
        points = points[:, :3]
        
        # Downsample to num_points if necessary
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad with random points if too few
            pad_size = self.num_points - len(points)
            pad_points = points[np.random.choice(len(points), pad_size, replace=True)]
            points = np.vstack([points, pad_points])
        
        return points


class NuScenesDataset(PointCloudDataset):
    """
    Dataset for nuScenes point cloud sequences.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 10,
                 prediction_length: int = 5,
                 num_points: int = 1024,
                 transforms: Optional[Any] = None,
                 split: str = 'train'):
        super(NuScenesDataset, self).__init__(
            data_path, sequence_length, prediction_length, num_points, transforms
        )
        self.split = split
        
    def _load_data_files(self) -> List[str]:
        """Load nuScenes data files."""
        data_files = []
        
        # nuScenes structure: data_path/sweeps/LIDAR_TOP/*.pcd.bin
        sweeps_dir = os.path.join(self.data_path, 'sweeps', 'LIDAR_TOP')
        
        if not os.path.exists(sweeps_dir):
            raise ValueError(f"nuScenes sweeps directory not found: {sweeps_dir}")
        
        # Get all .pcd.bin files
        bin_files = sorted([f for f in os.listdir(sweeps_dir) if f.endswith('.pcd.bin')])
        for bin_file in bin_files:
            data_files.append(os.path.join(sweeps_dir, bin_file))
        
        return data_files
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load nuScenes point cloud from .pcd.bin file."""
        # nuScenes .pcd.bin files contain float32 data with x, y, z, intensity
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
        # Take only x, y, z coordinates (ignore intensity)
        points = points[:, :3]
        
        # Downsample to num_points if necessary
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad with random points if too few
            pad_size = self.num_points - len(points)
            pad_points = points[np.random.choice(len(points), pad_size, replace=True)]
            points = np.vstack([points, pad_points])
        
        return points


class SyntheticDataset(PointCloudDataset):
    """
    Synthetic dataset for testing and development.
    Generates random point cloud sequences.
    """
    
    def __init__(self, 
                 num_sequences: int = 1000,
                 sequence_length: int = 10,
                 prediction_length: int = 5,
                 num_points: int = 1024,
                 transforms: Optional[Any] = None):
        self.num_sequences = num_sequences
        super(SyntheticDataset, self).__init__(
            "", sequence_length, prediction_length, num_points, transforms
        )
    
    def _load_data_files(self) -> List[str]:
        """Generate synthetic data file paths."""
        return [f"synthetic_{i}" for i in range(self.num_sequences)]
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Generate synthetic point cloud."""
        # Generate random point cloud with some structure
        points = np.random.randn(self.num_points, 3) * 10
        
        # Add some structure (e.g., ground plane)
        points[:, 2] = np.abs(points[:, 2])  # Make z positive (ground plane)
        
        # Add some temporal variation based on file index
        file_idx = int(file_path.split('_')[1])
        time_offset = file_idx * 0.1
        points[:, 0] += np.sin(time_offset) * 2  # Add sinusoidal motion
        points[:, 1] += np.cos(time_offset) * 2
        
        return points


class PointCloudSequenceDataset(data.Dataset):
    """
    Dataset that returns complete sequences instead of individual samples.
    Useful for sequence-to-sequence learning.
    """
    
    def __init__(self, 
                 base_dataset: PointCloudDataset,
                 sequence_stride: int = 1):
        super(PointCloudSequenceDataset, self).__init__()
        
        self.base_dataset = base_dataset
        self.sequence_stride = sequence_stride
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
    
    def _create_sequence_indices(self) -> List[List[int]]:
        """Create indices for complete sequences."""
        sequence_indices = []
        
        total_length = (self.base_dataset.sequence_length + 
                       self.base_dataset.prediction_length)
        
        for i in range(0, len(self.base_dataset) - total_length + 1, self.sequence_stride):
            sequence_indices.append(list(range(i, i + total_length)))
        
        return sequence_indices
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a complete sequence."""
        indices = self.sequence_indices[idx]
        
        # Load all point clouds in the sequence
        point_clouds = []
        for i in indices:
            sample = self.base_dataset[i]
            point_clouds.append(sample['input_sequence'])
        
        # Stack into sequence tensor
        input_sequence = torch.stack(point_clouds[:self.base_dataset.sequence_length])
        target_sequence = torch.stack(point_clouds[self.base_dataset.sequence_length:])
        
        return {
            'input_sequence': input_sequence,
            'target_sequence': target_sequence
        }
