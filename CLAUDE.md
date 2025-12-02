# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of 3D U-Net and its variants for medical image segmentation and regression tasks. The project supports both 2D and 3D image processing and provides pre-trained models for plant tissue segmentation.

## Architecture

### Core Modules

- `pytorch3dunet/unet3d/model.py` - Main U-Net model implementations (UNet3D, UNet3DFPGA, ResidualUNet3D, ResidualUNetSE3D)
- `pytorch3dunet/unet3d/trainer.py` - Training orchestration and model management
- `pytorch3dunet/unet3d/predictor.py` - Prediction pipeline for inference
- `pytorch3dunet/datasets/` - Data loading utilities (HDF5 datasets, transformations)
- `pytorch3dunet/augment/` - Data augmentation transforms
- `pytorch3dunet/unet3d/losses.py` - Loss functions for segmentation and regression
- `pytorch3dunet/unet3d/metrics.py` - Evaluation metrics

### Key Components

- **Models**: Standard 3D U-Net, Residual U-Net, and SE (Squeeze-Excite) variants
- **Data Format**: HDF5 files with `raw` and `label` datasets
- **Configuration**: YAML-based configuration system for training and prediction
- **Entry Points**: `train3dunet` and `predict3dunet` command-line tools

## Development Commands

### Installation
```bash
# Development installation
pip install -e .

# Or install from source
python setup.py install
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v tests/
```

### Training and Prediction
```bash
# Train a model
train3dunet --config path/to/config.yml

# Run prediction
predict3dunet --config path/to/test_config.yml

# Monitor training with TensorBoard
tensorboard --logdir <checkpoint_dir>/logs/
```

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yaml
conda activate 3dunet

# Or install via conda-forge
conda install -c conda-forge pytorch-3dunet
```

## Configuration System

The project uses YAML configuration files that specify:
- Model architecture and parameters
- Dataset paths and preprocessing
- Training hyperparameters
- Loss functions and metrics
- Data augmentation transforms

Example configurations are available in the `resources/` directory for different use cases (2D/3D segmentation, regression, multi-class).

## Data Requirements

- Input data must be in HDF5 format
- For training: files must contain both `raw` and `label` datasets
- For prediction: files must contain `raw` dataset
- Supported formats:
  - 2D single-channel: `(1, Y, X)`
  - 2D multi-channel: `(C, 1, Y, X)`
  - 3D single-channel: `(Z, Y, X)`
  - 3D multi-channel: `(C, Z, Y, X)`

## GPU Usage

The project automatically uses all available GPUs with DataParallel. To restrict GPU usage:
```bash
CUDA_VISIBLE_DEVICES=0,1 train3dunet --config config.yml
CUDA_VISIBLE_DEVICES=0,1 predict3dunet --config config.yml
```

## Testing Infrastructure

Tests use pytest with fixtures defined in `tests/conftest.py`:
- Mock HDF5 datasets for unit testing
- Configuration templates
- Random data generators

Test coverage includes:
- Model architecture validation
- Loss function correctness
- Data loading pipelines
- Training/prediction workflows
