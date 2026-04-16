#!/usr/bin/env python
"""
Example ONNX Inference Script

Demonstrates how to use the exported UNet3DFPGA ONNX model for inference.

Usage:
    python example_onnx_inference.py --model unet3dfpga.onnx --input sample_input.npy
"""

import argparse
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path


def load_onnx_model(model_path):
    """Load ONNX model and create inference session."""
    print(f"Loading ONNX model from {model_path}")

    # Create inference session
    session = ort.InferenceSession(model_path)

    # Get model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Model loaded successfully!")
    print(f"Input name: {input_info.name}")
    print(f"Input shape: {input_info.shape}")
    print(f"Input type: {input_info.type}")
    print(f"Output name: {output_info.name}")
    print(f"Output shape: {output_info.shape}")
    print(f"Output type: {output_info.type}")

    return session


def run_inference(session, input_data):
    """
    Run inference on the input data.

    Args:
        session: ONNX Runtime inference session
        input_data: Input numpy array of shape (batch, channels, depth, height, width)

    Returns:
        Output numpy array of shape (batch, out_channels, depth, height, width)
    """
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Ensure input is float32
    input_data = input_data.astype(np.float32)

    print(f"\nRunning inference...")
    print(f"Input shape: {input_data.shape}")

    # Run inference
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    inference_time = time.time() - start_time

    print(f"Inference completed in {inference_time*1000:.2f}ms")
    print(f"Output shape: {outputs[0].shape}")

    return outputs[0]


def create_dummy_input(batch_size=1, channels=1, depth=11, height=43, width=43):
    """Create dummy input data for testing."""
    print(f"\nCreating dummy input with shape ({batch_size}, {channels}, {depth}, {height}, {width})")
    input_data = np.random.randn(batch_size, channels, depth, height, width).astype(np.float32)
    return input_data


def analyze_output(output_data):
    """Analyze the model output."""
    print(f"\nOutput Analysis:")
    print(f"Shape: {output_data.shape}")
    print(f"Min value: {output_data.min():.6f}")
    print(f"Max value: {output_data.max():.6f}")
    print(f"Mean value: {output_data.mean():.6f}")
    print(f"Std value: {output_data.std():.6f}")

    # For softmax output, check if probabilities sum to 1 across channels
    if output_data.shape[1] > 1:  # Multi-class segmentation
        prob_sum = output_data.sum(axis=1)
        print(f"\nProbability sum across channels (should be ~1.0 for softmax):")
        print(f"Min: {prob_sum.min():.6f}, Max: {prob_sum.max():.6f}, Mean: {prob_sum.mean():.6f}")

    # Get predicted class for each pixel
    predicted_classes = np.argmax(output_data, axis=1)
    print(f"\nPredicted classes distribution:")
    for class_id in range(output_data.shape[1]):
        count = (predicted_classes == class_id).sum()
        percentage = (count / predicted_classes.size) * 100
        print(f"Class {class_id}: {count} pixels ({percentage:.2f}%)")


def save_output(output_data, output_path):
    """Save output to file."""
    output_path = Path(output_path)
    print(f"\nSaving output to {output_path}")
    np.save(output_path, output_data)
    print(f"Output saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Run ONNX inference on UNet3DFPGA model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the ONNX model file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input .npy file (if not provided, dummy input will be used)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output .npy file (optional)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (only used for dummy input)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run multiple iterations for benchmarking')
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='Number of iterations for benchmarking')

    args = parser.parse_args()

    # Load model
    session = load_onnx_model(args.model)

    # Prepare input
    if args.input:
        print(f"\nLoading input from {args.input}")
        input_data = np.load(args.input)
        print(f"Loaded input shape: {input_data.shape}")
    else:
        print("\nNo input file provided, using dummy input")
        input_data = create_dummy_input(batch_size=args.batch_size)

    # Run inference
    if args.benchmark:
        print(f"\n{'='*60}")
        print(f"Running benchmark with {args.num_iterations} iterations")
        print(f"{'='*60}")

        # Warmup
        print("Warming up...")
        for _ in range(3):
            _ = run_inference(session, input_data)

        # Benchmark
        inference_times = []
        for i in range(args.num_iterations):
            start = time.time()
            output = run_inference(session, input_data)
            inference_times.append(time.time() - start)

        # Statistics
        mean_time = np.mean(inference_times) * 1000
        std_time = np.std(inference_times) * 1000
        min_time = np.min(inference_times) * 1000
        max_time = np.max(inference_times) * 1000

        print(f"\n{'='*60}")
        print(f"Benchmark Results ({args.num_iterations} iterations)")
        print(f"{'='*60}")
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Throughput: {1000/mean_time:.2f} inferences/second")
    else:
        output = run_inference(session, input_data)

        # Analyze output
        analyze_output(output)

        # Save output if requested
        if args.output:
            save_output(output, args.output)


if __name__ == '__main__':
    main()
