#!/usr/bin/env python
"""
ONNX Export Script for UNet3DFPGA Model

This script exports a trained UNet3DFPGA model to ONNX format for deployment and inference.

Usage:
    python onnx_export.py --config resources/ecal/train_config_128_mixed.yaml \\
                          --checkpoint resources/ecal/checkpoint_128_mixed/best_checkpoint.pytorch \\
                          --output model.onnx

Arguments:
    --config: Path to the training configuration YAML file
    --checkpoint: Path to the model checkpoint file
    --output: Output path for the ONNX model (default: model.onnx)
    --batch-size: Batch size for ONNX export (default: 1)
    --opset-version: ONNX opset version (default: 11)
"""

import argparse
import torch
import yaml
import logging
from pathlib import Path

from pytorch3dunet.unet3d.model import get_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config_path, checkpoint_path):
    """Load the model from configuration and checkpoint."""
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Get model configuration
    model_config = config['model']
    logger.info(f"Model configuration: {model_config}")

    # Create model
    logger.info(f"Creating model: {model_config['name']}")
    model = get_model(model_config)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle DataParallel wrapped models
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")

    return model, model_config


def export_to_onnx(model, model_config, output_path, batch_size=1, opset_version=11):
    """
    Export the model to ONNX format.

    Args:
        model: The PyTorch model to export
        model_config: Model configuration dictionary
        output_path: Path to save the ONNX model
        batch_size: Batch size for the dummy input
        opset_version: ONNX opset version
    """
    # Get input configuration
    in_channels = model_config['in_channels']
    depth_channels = model_config.get('depth_channels', 11)

    # For UNet3DFPGA, the input shape is (batch, channels, depth, height, width)
    # From the config: patch_shape: [43, 43, 11] which is [H, W, D]
    # Model expects (batch, C, D, H, W)
    height, width = 43, 43

    # Create dummy input
    dummy_input = torch.randn(batch_size, in_channels, depth_channels, height, width)

    logger.info(f"Dummy input shape: {dummy_input.shape}")
    logger.info(f"Exporting model to ONNX format...")
    logger.info(f"Output path: {output_path}")
    logger.info(f"ONNX opset version: {opset_version}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        #dynamic_axes={
        #    'input': {0: 'batch_size', 3: 'height', 4: 'width'},
        #    'output': {0: 'batch_size', 3: 'height', 4: 'width'}
        #}
    )

    logger.info(f"Model successfully exported to {output_path}")

    # Verify the exported model
    try:
        import onnx
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed!")

        # Print model info
        logger.info(f"ONNX model inputs: {[input.name for input in onnx_model.graph.input]}")
        logger.info(f"ONNX model outputs: {[output.name for output in onnx_model.graph.output]}")

    except ImportError:
        logger.warning("onnx package not found. Skipping model verification.")
        logger.warning("Install with: pip install onnx")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        raise


def test_onnx_inference(onnx_path, dummy_input):
    """
    Test ONNX model inference using onnxruntime.

    Args:
        onnx_path: Path to the ONNX model
        dummy_input: Dummy input tensor for testing
    """
    try:
        import onnxruntime as ort

        logger.info("Testing ONNX inference with onnxruntime...")

        # Create inference session
        ort_session = ort.InferenceSession(onnx_path)

        # Get input/output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        # Run inference
        ort_inputs = {input_name: dummy_input.numpy()}
        ort_outputs = ort_session.run([output_name], ort_inputs)

        logger.info(f"ONNX inference output shape: {ort_outputs[0].shape}")
        logger.info("ONNX inference test passed!")

    except ImportError:
        logger.warning("onnxruntime package not found. Skipping inference test.")
        logger.warning("Install with: pip install onnxruntime")
    except Exception as e:
        logger.error(f"ONNX inference test failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Export UNet3DFPGA model to ONNX format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the training configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output path for the ONNX model')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for ONNX export')
    parser.add_argument('--opset-version', type=int, default=18,
                        help='ONNX opset version (default: 18, recommended for modern PyTorch)')
    parser.add_argument('--test-inference', action='store_true',
                        help='Test ONNX inference after export')

    args = parser.parse_args()

    # Validate paths
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, model_config = load_model(args.config, args.checkpoint)

    # Export to ONNX
    export_to_onnx(
        model,
        model_config,
        str(output_path),
        batch_size=args.batch_size,
        opset_version=args.opset_version
    )

    # Test inference if requested
    if args.test_inference:
        in_channels = model_config['in_channels']
        depth_channels = model_config.get('depth_channels', 11)
        dummy_input = torch.randn(args.batch_size, in_channels, depth_channels, 43, 43)
        test_onnx_inference(str(output_path), dummy_input)

    logger.info("Export completed successfully!")


if __name__ == '__main__':
    main()
