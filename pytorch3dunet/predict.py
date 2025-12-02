import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.fpga_unet_modular import UNet3DFPGAModular

logger = utils.get_logger('UNet3DPredict')


def get_predictor(model, config):
    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)
    out_channels = config['model'].get('out_channels')
    return predictor_class(model, output_dir, out_channels, **predictor_config)


def load_model_checkpoint(model, model_path):
    """
    Load model checkpoint with support for modular FPGA checkpoints.

    Args:
        model: The model instance to load checkpoints into
        model_path: Path to the checkpoint file or directory
    """
    # Check if this is a UNet3DFPGAModular model
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    if isinstance(actual_model, UNet3DFPGAModular):
        # Check if model_path is a directory containing modular checkpoints
        modular_dir = os.path.join(os.path.dirname(model_path), 'modular_blocks')
        metadata_path = os.path.join(modular_dir, 'model_metadata.pth')

        if os.path.exists(metadata_path):
            logger.info(f'Loading modular checkpoints from {modular_dir}...')
            actual_model.load_modular_checkpoints(modular_dir)
            return
        else:
            logger.info(f'Modular checkpoint directory not found at {modular_dir}, trying standard checkpoint...')

    # Fall back to standard checkpoint loading
    logger.info(f'Loading standard checkpoint from {model_path}...')
    utils.load_checkpoint(model_path, model)


def main():
    # Load configuration
    config, _ = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    load_model_checkpoint(model, model_path)

    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    # create predictor instance
    predictor = get_predictor(model, config)

    metrics = []
    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        metric = predictor(test_loader)
        if metric is not None:
            metrics.append(metric)

    if metrics:
        # average across loaders
        metrics = torch.Tensor(metrics)
        per_class_metrics = metrics.mean(dim=0)
        avg_metric = metrics.mean()
        logger.info(f'Per-class average metric: {per_class_metrics}')
        logger.info(f'Average metric: {avg_metric}')


if __name__ == '__main__':
    main()
