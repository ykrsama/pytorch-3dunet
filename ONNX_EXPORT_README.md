# ONNX Export for UNet3DFPGA

This guide explains how to export the UNet3DFPGA model to ONNX format for deployment.

## Prerequisites

```bash
pip install onnx onnxruntime
```

## Basic Usage

Export the trained model to ONNX:

```bash
python onnx_export.py \
    --config resources/ecal/train_config_128_mixed.yaml \
    --checkpoint resources/ecal/checkpoint_128_mixed/best_checkpoint.pytorch \
    --output unet3dfpga.onnx
```

## Command-Line Arguments

- `--config`: Path to the training configuration YAML file (required)
- `--checkpoint`: Path to the model checkpoint file (required)
- `--output`: Output path for the ONNX model (default: `model.onnx`)
- `--batch-size`: Batch size for ONNX export (default: 1)
- `--opset-version`: ONNX opset version (default: 11)
- `--test-inference`: Test ONNX inference after export (optional)

## Examples

### Export with default settings

```bash
python onnx_export.py \
    --config resources/ecal/train_config_128_mixed.yaml \
    --checkpoint resources/ecal/checkpoint_128_mixed/best_checkpoint.pytorch
```

### Export with custom batch size and test inference

```bash
python onnx_export.py \
    --config resources/ecal/train_config_128_mixed.yaml \
    --checkpoint resources/ecal/checkpoint_128_mixed/best_checkpoint.pytorch \
    --output models/unet3dfpga_batch4.onnx \
    --batch-size 4 \
    --test-inference
```

### Export with specific ONNX opset version

```bash
python onnx_export.py \
    --config resources/ecal/train_config_128_mixed.yaml \
    --checkpoint resources/ecal/checkpoint_128_mixed/best_checkpoint.pytorch \
    --output unet3dfpga_opset13.onnx \
    --opset-version 13
```

## Model Details

### UNet3DFPGA Architecture

The UNet3DFPGA model is an FPGA-optimized variant with a simplified architecture:

- **Input Shape**: `(batch, channels, depth, height, width)`
  - For the provided config: `(batch, 1, 11, 43, 43)`
  - channels: 1 (from config `in_channels`)
  - depth: 11 (from config `depth_channels`)
  - height, width: 43 (from config `patch_shape`)

- **Output Shape**: `(batch, out_channels, depth, height, width)`
  - For the provided config: `(batch, 5, 11, 43, 43)`
  - out_channels: 5 (from config `out_channels`)

- **Feature Maps**: `[128, 256]` (encoder levels)
- **Activation**: Softmax (since `final_sigmoid: false`)

### Dynamic Axes

The ONNX model supports dynamic batch size, height, and width:

- `batch_size`: Dynamic
- `height`: Dynamic (spatial dimension)
- `width`: Dynamic (spatial dimension)
- `depth`: Fixed at 11 (as per model design)
- `channels`: Fixed at 1

## Using the ONNX Model

### Python with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("unet3dfpga.onnx")

# Prepare input (example)
input_data = np.random.randn(1, 1, 11, 43, 43).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})

print(f"Output shape: {output[0].shape}")
```

### Verifying the ONNX Model

```python
import onnx

# Load and check the model
model = onnx.load("unet3dfpga.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid!")

# Print model info
print("Inputs:")
for input in model.graph.input:
    print(f"  {input.name}: {input.type}")

print("Outputs:")
for output in model.graph.output:
    print(f"  {output.name}: {output.type}")
```

## Troubleshooting

### DataParallel Models

The script automatically handles models saved with `torch.nn.DataParallel` by removing the `module.` prefix from state dict keys.

### Memory Issues

If you encounter memory issues during export:
- Reduce `--batch-size` to 1
- Ensure you have sufficient RAM (the model loads entirely into memory)

### ONNX Opset Version

Different deployment targets may require different ONNX opset versions:
- ONNX Runtime: Generally supports opset 7-15
- TensorRT: Check your TensorRT version for supported opsets
- CoreML: Typically supports opset 11-13

Use `--opset-version` to specify the version needed for your target platform.

## Model Information

The exported ONNX model includes:

- All trained weights and biases
- Model architecture (conv layers, pooling, upsampling, etc.)
- Activation functions (ReLU, Softmax)
- Batch normalization layers (if present)

## Performance Considerations

- The model is optimized for 2D operations (Conv2d, MaxPool2d)
- Input depth is handled by reshaping: `(B, C, D, H, W)` → `(B, C*D, H, W)`
- Output is reshaped back: `(B, out*D, H, W)` → `(B, out, D, H, W)`
- This design is FPGA-friendly and enables efficient hardware implementation

## Further Deployment

After exporting to ONNX, you can:

1. **Optimize for inference**: Use `onnxruntime` optimization tools
2. **Convert to other formats**:
   - TensorRT (NVIDIA GPUs)
   - OpenVINO (Intel hardware)
   - CoreML (Apple devices)
3. **Quantize**: Convert to INT8 for faster inference
4. **Deploy**: Use in production with ONNX Runtime or other inference engines
