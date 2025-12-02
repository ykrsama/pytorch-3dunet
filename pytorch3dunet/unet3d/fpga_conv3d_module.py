import torch
import torch.nn as nn
from .buildingblocks import DoubleConv


class FPGAConv3DBlock(nn.Module):
    """
    Modular Conv3D block designed for HLS compilation compatibility.
    This module encapsulates a single convolutional block that can be compiled
    and deployed independently on FPGA hardware.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        block_id (str): unique identifier for this block (used for checkpoint naming)
        kernel_size (int): size of the convolving kernel, default: 3
        layer_order (str): determines the order of layers, default: 'gcr'
        num_groups (int): number of groups for GroupNorm, default: 8
        padding (int): padding for convolution, default: 1
        dropout_prob (float): dropout probability, default: 0.1
        is_encoder (bool): whether this block is in encoder path, default: True
    """

    def __init__(self, in_channels, out_channels, block_id, kernel_size=3,
                 layer_order='gcr', num_groups=8, padding=1, dropout_prob=0.1,
                 is_encoder=True):
        super(FPGAConv3DBlock, self).__init__()

        self.block_id = block_id
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Use DoubleConv as the base building block
        self.conv_block = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder=is_encoder,
            kernel_size=kernel_size,
            order=layer_order,
            num_groups=num_groups,
            padding=padding,
            dropout_prob=dropout_prob,
            is3d=True
        )

    def forward(self, x):
        """
        Forward pass through the conv3d block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor after convolution operations
        """
        return self.conv_block(x)

    def save_checkpoint(self, filepath):
        """
        Save this specific block's parameters to a checkpoint file.

        Args:
            filepath (str): Path to save the checkpoint
        """
        checkpoint = {
            'block_id': self.block_id,
            'state_dict': self.state_dict(),
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load this specific block's parameters from a checkpoint file.

        Args:
            filepath (str): Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint

    def get_block_info(self):
        """
        Get information about this block for HLS compilation.

        Returns:
            dict: Block information including dimensions and parameters
        """
        return {
            'block_id': self.block_id,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class FPGAInputConv3DBlock(FPGAConv3DBlock):
    """
    Specialized input conv3d block for UNet3DFPGA input layer.
    This is the first block that processes the raw input.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FPGAInputConv3DBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_id="input_conv",
            is_encoder=True,
            **kwargs
        )


class FPGAEncoderConv3DBlock(FPGAConv3DBlock):
    """
    Specialized encoder conv3d block for UNet3DFPGA encoder layer.
    This block processes the downsampled features.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FPGAEncoderConv3DBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_id="encoder_conv",
            is_encoder=True,
            **kwargs
        )


class FPGADecoderConv3DBlock(FPGAConv3DBlock):
    """
    Specialized decoder conv3d block for UNet3DFPGA decoder layer.
    This block processes the upsampled and concatenated features.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FPGADecoderConv3DBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_id="decoder_conv",
            is_encoder=False,
            **kwargs
        )


class FPGAOutputConv3DBlock(FPGAConv3DBlock):
    """
    Specialized output conv3d block for UNet3DFPGA output layer.
    This is the final conv block before the 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(FPGAOutputConv3DBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            block_id="output_conv",
            is_encoder=False,
            **kwargs
        )