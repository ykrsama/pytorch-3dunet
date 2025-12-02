import torch
import torch.nn as nn
import os

from .fpga_conv3d_module import (
    FPGAInputConv3DBlock,
    FPGAEncoderConv3DBlock,
    FPGADecoderConv3DBlock,
    FPGAOutputConv3DBlock
)
from .fpga_pooling_module import (
    FPGAMaxPoolBlock,
    FPGANearestUpsampleBlock
)


class UNet3DFPGAModular(nn.Module):
    """
    Modular FPGA-optimized 3D U-Net with separated conv3d and pooling blocks.

    This version separates all conv3d and pooling operations into individual
    modules that can be compiled separately for HLS implementation while
    maintaining the same architecture and functionality as the original UNet3DFPGA.

    Architecture:
    input → InputConv → f_maps[0] ─── concat ──→ DecoderConv → f_maps[0] → OutputConv → output
                          │                         ↑
                      MaxPool                  Upsampling
                          ↓                         │
                   EncoderConv → f_maps[1] ────────┘

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        final_sigmoid (bool): if True apply Sigmoid activation, default: True
        f_maps (list): feature map dimensions [encoder_out, decoder_out], default: [64, 128]
        layer_order (str): layer order for conv blocks, default: 'gcr'
        num_groups (int): number of groups for GroupNorm, default: 8
        is_segmentation (bool): if True apply final activation, default: True
        conv_padding (int): padding for convolutions, default: 1
        dropout_prob (float): dropout probability, default: 0.1
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=[64, 128],
                 layer_order='gcr', num_groups=8, is_segmentation=True, conv_padding=1,
                 dropout_prob=0.1, **kwargs):
        super(UNet3DFPGAModular, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps, f_maps * 2]

        assert len(f_maps) == 2, "FPGA-optimized U-Net requires exactly 2 feature map levels"

        self.f_maps = f_maps
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Modular blocks - each can be compiled separately for HLS
        self.input_conv = FPGAInputConv3DBlock(
            in_channels=in_channels,
            out_channels=f_maps[0],
            kernel_size=3,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=conv_padding,
            dropout_prob=dropout_prob
        )

        self.encoder_pool = FPGAMaxPoolBlock(kernel_size=2)

        self.encoder_conv = FPGAEncoderConv3DBlock(
            in_channels=f_maps[0],
            out_channels=f_maps[1],
            kernel_size=3,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=conv_padding,
            dropout_prob=dropout_prob
        )

        self.decoder_upsample = FPGANearestUpsampleBlock(scale_factor=2)

        self.decoder_conv = FPGADecoderConv3DBlock(
            in_channels=f_maps[0] + f_maps[1],  # concatenated features
            out_channels=f_maps[0],
            kernel_size=3,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=conv_padding,
            dropout_prob=dropout_prob
        )

        self.output_conv = FPGAOutputConv3DBlock(
            in_channels=f_maps[0],
            out_channels=f_maps[0],
            kernel_size=3,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=conv_padding,
            dropout_prob=dropout_prob
        )

        # Final 1x1 convolution (can be treated as a separate block if needed)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)

        # Final activation
        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x, return_logits=False):
        """
        Forward pass through the modular U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)
            return_logits (bool): If True, return both output and logits

        Returns:
            torch.Tensor or tuple: Output tensor, optionally with logits
        """
        # Input → Conv+ReLU → f_maps[0] (skip connection)
        x1 = self.input_conv(x)

        # MaxPool → f_maps[0] → Conv+ReLU → f_maps[1]
        x_pooled = self.encoder_pool(x1)
        x2 = self.encoder_conv(x_pooled)

        # Upsampling f_maps[1] → f_maps[0] spatial dimensions
        x_up = self.decoder_upsample(x2)

        # Ensure upsampled tensor matches skip connection size
        if x_up.shape[2:] != x1.shape[2:]:
            x_up = self.decoder_upsample(x2, target_size=x1.shape[2:])

        # Concatenate skip connection: concat(x1, x_up) → f_maps[0] + f_maps[1]
        x_concat = torch.cat([x1, x_up], dim=1)

        # Decoder convolution: (f_maps[0] + f_maps[1]) → Conv+ReLU → f_maps[0]
        x_dec = self.decoder_conv(x_concat)

        # Output convolution: f_maps[0] → Conv+ReLU → f_maps[0]
        x_out = self.output_conv(x_dec)

        # Final 1x1 convolution: f_maps[0] → out_channels
        logits = self.final_conv(x_out)

        if self.final_activation is not None:
            output = self.final_activation(logits)
            if return_logits:
                return output, logits
            return output

        if return_logits:
            return logits, logits
        return logits

    def save_modular_checkpoints(self, checkpoint_dir):
        """
        Save individual checkpoints for each modular block.

        Args:
            checkpoint_dir (str): Directory to save individual checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save each modular block separately
        blocks = {
            'input_conv': self.input_conv,
            'encoder_pool': self.encoder_pool,
            'encoder_conv': self.encoder_conv,
            'decoder_upsample': self.decoder_upsample,
            'decoder_conv': self.decoder_conv,
            'output_conv': self.output_conv
        }

        checkpoint_paths = {}
        for block_name, block in blocks.items():
            checkpoint_path = os.path.join(checkpoint_dir, f"{block_name}.pth")
            block.save_checkpoint(checkpoint_path)
            checkpoint_paths[block_name] = checkpoint_path

        # Save final conv layer separately (not modular block)
        final_conv_path = os.path.join(checkpoint_dir, "final_conv.pth")
        torch.save({
            'state_dict': self.final_conv.state_dict(),
            'in_features': self.f_maps[0],
            'out_features': self.out_channels,
            'model_class': 'Conv3d'
        }, final_conv_path)
        checkpoint_paths['final_conv'] = final_conv_path

        # Save model metadata
        metadata_path = os.path.join(checkpoint_dir, "model_metadata.pth")
        torch.save({
            'model_class': self.__class__.__name__,
            'f_maps': self.f_maps,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'checkpoint_paths': checkpoint_paths
        }, metadata_path)

        return checkpoint_paths

    def load_modular_checkpoints(self, checkpoint_dir):
        """
        Load individual checkpoints for each modular block.

        Args:
            checkpoint_dir (str): Directory containing individual checkpoints
        """
        # Load each modular block
        self.input_conv.load_checkpoint(
            os.path.join(checkpoint_dir, "input_conv.pth"))
        self.encoder_pool.load_checkpoint(
            os.path.join(checkpoint_dir, "encoder_pool.pth"))
        self.encoder_conv.load_checkpoint(
            os.path.join(checkpoint_dir, "encoder_conv.pth"))
        self.decoder_upsample.load_checkpoint(
            os.path.join(checkpoint_dir, "decoder_upsample.pth"))
        self.decoder_conv.load_checkpoint(
            os.path.join(checkpoint_dir, "decoder_conv.pth"))
        self.output_conv.load_checkpoint(
            os.path.join(checkpoint_dir, "output_conv.pth"))

        # Load final conv layer
        final_conv_checkpoint = torch.load(
            os.path.join(checkpoint_dir, "final_conv.pth"),
            map_location='cpu'
        )
        self.final_conv.load_state_dict(final_conv_checkpoint['state_dict'])

    def get_modular_blocks_info(self):
        """
        Get information about all modular blocks for HLS compilation.

        Returns:
            dict: Information about each modular block
        """
        return {
            'input_conv': self.input_conv.get_block_info(),
            'encoder_pool': self.encoder_pool.get_block_info(),
            'encoder_conv': self.encoder_conv.get_block_info(),
            'decoder_upsample': self.decoder_upsample.get_block_info(),
            'decoder_conv': self.decoder_conv.get_block_info(),
            'output_conv': self.output_conv.get_block_info(),
            'final_conv': {
                'block_id': 'final_conv',
                'in_channels': self.f_maps[0],
                'out_channels': self.out_channels,
                'total_params': sum(p.numel() for p in self.final_conv.parameters()),
                'trainable_params': sum(p.numel() for p in self.final_conv.parameters() if p.requires_grad)
            }
        }

    def get_hls_compilation_order(self):
        """
        Get the recommended order for HLS compilation of blocks.

        Returns:
            list: Ordered list of block names for compilation
        """
        return [
            'input_conv',
            'encoder_pool',
            'encoder_conv',
            'decoder_upsample',
            'decoder_conv',
            'output_conv',
            'final_conv'
        ]