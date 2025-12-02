import torch
import torch.nn as nn
import torch.nn.functional as F


class FPGAPoolingBlock(nn.Module):
    """
    Modular pooling block designed for HLS compilation compatibility.
    This module encapsulates pooling operations that can be compiled
    and deployed independently on FPGA hardware.

    Args:
        kernel_size (int or tuple): size of the pooling kernel, default: 2
        stride (int or tuple): stride of the pooling operation, default: None (same as kernel_size)
        pool_type (str): type of pooling ('max' or 'avg'), default: 'max'
        block_id (str): unique identifier for this block
    """

    def __init__(self, kernel_size=2, stride=None, pool_type='max', block_id='pool'):
        super(FPGAPoolingBlock, self).__init__()

        self.block_id = block_id
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pool_type = pool_type

        if pool_type == 'max':
            self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=self.stride)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=kernel_size, stride=self.stride)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Must be 'max' or 'avg'")

    def forward(self, x):
        """
        Forward pass through the pooling block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor after pooling operation
        """
        return self.pool(x)

    def save_checkpoint(self, filepath):
        """
        Save this specific block's configuration to a checkpoint file.

        Args:
            filepath (str): Path to save the checkpoint
        """
        checkpoint = {
            'block_id': self.block_id,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'pool_type': self.pool_type,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load this specific block's configuration from a checkpoint file.
        Note: Pooling layers don't have learnable parameters, so this mainly
        loads configuration for consistency.

        Args:
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint

    def get_block_info(self):
        """
        Get information about this block for HLS compilation.

        Returns:
            dict: Block information including configuration
        """
        return {
            'block_id': self.block_id,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'pool_type': self.pool_type,
            'total_params': 0,  # Pooling layers have no parameters
            'trainable_params': 0
        }


class FPGAUpsamplingBlock(nn.Module):
    """
    Modular upsampling block designed for HLS compilation compatibility.
    This replaces nn.Upsample with custom interpolation logic that can be
    more easily adapted for FPGA implementation.

    Args:
        scale_factor (int or tuple): multiplier for spatial size, default: 2
        mode (str): upsampling algorithm, default: 'nearest'
        block_id (str): unique identifier for this block
    """

    def __init__(self, scale_factor=2, mode='nearest', block_id='upsample'):
        super(FPGAUpsamplingBlock, self).__init__()

        self.block_id = block_id
        self.scale_factor = scale_factor
        self.mode = mode

        # Store supported modes for HLS compatibility
        self.supported_modes = ['nearest']  # Start with nearest neighbor only
        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} not supported for FPGA. Supported modes: {self.supported_modes}")

    def forward(self, x, target_size=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)
            target_size (tuple, optional): Target spatial size (D, H, W)

        Returns:
            torch.Tensor: Upsampled output tensor
        """
        if target_size is not None:
            return F.interpolate(x, size=target_size, mode=self.mode)
        else:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

    def save_checkpoint(self, filepath):
        """
        Save this specific block's configuration to a checkpoint file.

        Args:
            filepath (str): Path to save the checkpoint
        """
        checkpoint = {
            'block_id': self.block_id,
            'scale_factor': self.scale_factor,
            'mode': self.mode,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load this specific block's configuration from a checkpoint file.

        Args:
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint

    def get_block_info(self):
        """
        Get information about this block for HLS compilation.

        Returns:
            dict: Block information including configuration
        """
        return {
            'block_id': self.block_id,
            'scale_factor': self.scale_factor,
            'mode': self.mode,
            'total_params': 0,  # Upsampling layers have no parameters
            'trainable_params': 0
        }


class FPGAMaxPoolBlock(FPGAPoolingBlock):
    """
    Specialized MaxPool block for UNet3DFPGA encoder.
    """

    def __init__(self, kernel_size=2, **kwargs):
        super(FPGAMaxPoolBlock, self).__init__(
            kernel_size=kernel_size,
            pool_type='max',
            block_id='encoder_maxpool',
            **kwargs
        )


class FPGANearestUpsampleBlock(FPGAUpsamplingBlock):
    """
    Specialized nearest neighbor upsampling block for UNet3DFPGA decoder.
    """

    def __init__(self, scale_factor=2, **kwargs):
        super(FPGANearestUpsampleBlock, self).__init__(
            scale_factor=scale_factor,
            mode='nearest',
            block_id='decoder_upsample',
            **kwargs
        )