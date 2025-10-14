# S2Net: Stochastic Sequential Pointcloud Forecasting

This repository contains a PyTorch implementation of S2Net, a conditional variational recurrent neural network for predicting future LiDAR point clouds in autonomous driving scenarios.

## Features

- Conditional Variational Recurrent Neural Network (CVRNN) with temporally-dependent latent variables
- Pyramid-LSTM architecture with skip connections for enhanced prediction fidelity
- CUDA support for efficient training and inference
- Support for KITTI and nuScenes datasets
- Comprehensive evaluation metrics including Chamfer Distance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd S2Net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure CUDA is properly installed and compatible with your PyTorch version.

## Usage

### Training
```bash
python train.py --config configs/default.yaml
```

### Evaluation
```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_path data/test/
```

## Model Architecture

The S2Net model consists of:
- **Encoder**: LSTM-based encoder for processing sequential point cloud data
- **Variational Module**: Conditional variational component with temporally-dependent latent variables
- **Pyramid-LSTM**: Multi-level LSTM structure with skip connections
- **Decoder**: Point cloud reconstruction from latent representations

## Datasets

The model supports:
- KITTI dataset
- nuScenes dataset

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{s2net2022,
  title={S2Net: Stochastic Sequential Pointcloud Forecasting},
  author={[Authors]},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
