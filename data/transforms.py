import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import random


class PointCloudTransforms:
    """
    Collection of point cloud transformations for data augmentation.
    """
    
    def __init__(self, 
                 transforms: List[str] = None,
                 rotation_range: Tuple[float, float] = (-np.pi, np.pi),
                 translation_range: Tuple[float, float] = (-1.0, 1.0),
                 scaling_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 dropout_prob: float = 0.1):
        """
        Initialize transforms.
        
        Args:
            transforms: List of transform names to apply
            rotation_range: Range for random rotation (min, max) in radians
            translation_range: Range for random translation (min, max) in meters
            scaling_range: Range for random scaling (min, max)
            noise_std: Standard deviation for Gaussian noise
            dropout_prob: Probability of dropping points
        """
        self.transforms = transforms or ['rotation', 'translation', 'scaling', 'noise']
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        """
        Apply transforms to point cloud sequence.
        
        Args:
            point_clouds: Point cloud sequence [seq_len, num_points, 3]
            
        Returns:
            Transformed point clouds
        """
        transformed_pc = point_clouds.copy()
        
        for transform_name in self.transforms:
            if transform_name == 'rotation':
                transformed_pc = self._apply_rotation(transformed_pc)
            elif transform_name == 'translation':
                transformed_pc = self._apply_translation(transformed_pc)
            elif transform_name == 'scaling':
                transformed_pc = self._apply_scaling(transformed_pc)
            elif transform_name == 'noise':
                transformed_pc = self._apply_noise(transformed_pc)
            elif transform_name == 'dropout':
                transformed_pc = self._apply_dropout(transformed_pc)
            elif transform_name == 'normalization':
                transformed_pc = self._apply_normalization(transformed_pc)
        
        return transformed_pc
    
    def _apply_rotation(self, point_clouds: np.ndarray) -> np.ndarray:
        """Apply random rotation around Z-axis."""
        angle = random.uniform(*self.rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        return np.matmul(point_clouds, rotation_matrix.T)
    
    def _apply_translation(self, point_clouds: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        translation = np.array([
            random.uniform(*self.translation_range),
            random.uniform(*self.translation_range),
            random.uniform(*self.translation_range) * 0.1  # Smaller Z translation
        ])
        
        return point_clouds + translation
    
    def _apply_scaling(self, point_clouds: np.ndarray) -> np.ndarray:
        """Apply random scaling."""
        scale = random.uniform(*self.scaling_range)
        return point_clouds * scale
    
    def _apply_noise(self, point_clouds: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, point_clouds.shape)
        return point_clouds + noise
    
    def _apply_dropout(self, point_clouds: np.ndarray) -> np.ndarray:
        """Randomly drop some points."""
        if random.random() < self.dropout_prob:
            seq_len, num_points, _ = point_clouds.shape
            keep_indices = np.random.choice(
                num_points, 
                int(num_points * (1 - self.dropout_prob)), 
                replace=False
            )
            point_clouds = point_clouds[:, keep_indices, :]
            
            # Pad back to original size
            pad_size = num_points - len(keep_indices)
            if pad_size > 0:
                pad_points = point_clouds[:, np.random.choice(len(keep_indices), pad_size), :]
                point_clouds = np.concatenate([point_clouds, pad_points], axis=1)
        
        return point_clouds
    
    def _apply_normalization(self, point_clouds: np.ndarray) -> np.ndarray:
        """Normalize point clouds to unit sphere."""
        # Center the point clouds
        centroid = np.mean(point_clouds, axis=1, keepdims=True)
        point_clouds = point_clouds - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(point_clouds, axis=2), axis=1, keepdims=True)
        point_clouds = point_clouds / (max_dist[:, :, np.newaxis] + 1e-8)
        
        return point_clouds


class RandomRotation:
    """Random rotation transform."""
    
    def __init__(self, rotation_range: Tuple[float, float] = (-np.pi, np.pi)):
        self.rotation_range = rotation_range
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        angle = random.uniform(*self.rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        return np.matmul(point_clouds, rotation_matrix.T)


class RandomTranslation:
    """Random translation transform."""
    
    def __init__(self, translation_range: Tuple[float, float] = (-1.0, 1.0)):
        self.translation_range = translation_range
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        translation = np.array([
            random.uniform(*self.translation_range),
            random.uniform(*self.translation_range),
            random.uniform(*self.translation_range) * 0.1
        ])
        
        return point_clouds + translation


class RandomScaling:
    """Random scaling transform."""
    
    def __init__(self, scaling_range: Tuple[float, float] = (0.8, 1.2)):
        self.scaling_range = scaling_range
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        scale = random.uniform(*self.scaling_range)
        return point_clouds * scale


class GaussianNoise:
    """Gaussian noise transform."""
    
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, point_clouds.shape)
        return point_clouds + noise


class PointDropout:
    """Random point dropout transform."""
    
    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        if random.random() < self.dropout_prob:
            seq_len, num_points, _ = point_clouds.shape
            keep_indices = np.random.choice(
                num_points, 
                int(num_points * (1 - self.dropout_prob)), 
                replace=False
            )
            point_clouds = point_clouds[:, keep_indices, :]
            
            # Pad back to original size
            pad_size = num_points - len(keep_indices)
            if pad_size > 0:
                pad_points = point_clouds[:, np.random.choice(len(keep_indices), pad_size), :]
                point_clouds = np.concatenate([point_clouds, pad_points], axis=1)
        
        return point_clouds


class NormalizePointClouds:
    """Normalize point clouds to unit sphere."""
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        # Center the point clouds
        centroid = np.mean(point_clouds, axis=1, keepdims=True)
        point_clouds = point_clouds - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(point_clouds, axis=2), axis=1, keepdims=True)
        point_clouds = point_clouds / (max_dist + 1e-8)
        
        return point_clouds


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, point_clouds: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            point_clouds = transform(point_clouds)
        return point_clouds


def get_default_transforms(is_training: bool = True) -> PointCloudTransforms:
    """
    Get default transforms for training or validation.
    
    Args:
        is_training: Whether in training mode
        
    Returns:
        PointCloudTransforms object
    """
    if is_training:
        return PointCloudTransforms(
            transforms=['rotation', 'translation', 'scaling', 'noise'],
            rotation_range=(-np.pi/4, np.pi/4),
            translation_range=(-0.5, 0.5),
            scaling_range=(0.9, 1.1),
            noise_std=0.005,
            dropout_prob=0.05
        )
    else:
        return PointCloudTransforms(
            transforms=['normalization'],
            noise_std=0.0,
            dropout_prob=0.0
        )
