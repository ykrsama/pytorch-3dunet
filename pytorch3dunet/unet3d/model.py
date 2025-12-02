import torch
from torch import nn

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level
from pytorch3dunet.unet3d.fpga_unet_modular import UNet3DFPGAModular


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x, return_logits=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) for 3D or (N, C, H, W) for 2D,
                              where N is the batch size, C is the number of channels,
                              D is the depth, H is the height, and W is the width.
            return_logits (bool): If True, returns both the output and the logits.
                                  If False, returns only the output. Default is False.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
                          If return_logits is True, returns a tuple of (output, logits).
        """
        output, logits = self._forward_logits(x)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            # compute final activation
            out = self.final_activation(x)
            # return both probabilities and logits
            return out, x

        return x, x


class UNet3DFPGA(nn.Module):
    """
    FPGA-optimized 3D U-Net with reduced contraction path for efficient hardware implementation.

    Architecture:
    input → Conv+ReLU → f_maps[0] ─── concat ──→ f_maps[1] → Conv+ReLU → f_maps[0] → Conv+ReLU → output
                           │                         ↑
                       MaxPool                  Upsampling
                           ↓                         │
                    f_maps[0] → Conv+ReLU → f_maps[1]
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=[64, 128],
                 layer_order='gcr', num_groups=8, is_segmentation=True, conv_padding=1,
                 dropout_prob=0.1, **kwargs):
        super(UNet3DFPGA, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps, f_maps * 2]

        assert len(f_maps) == 2, "FPGA-optimized U-Net requires exactly 2 feature map levels"

        self.f_maps = f_maps

        # Input convolution: input → Conv+ReLU → f_maps[0]
        self.input_conv = DoubleConv(in_channels, f_maps[0], encoder=True,
                                     order=layer_order, num_groups=num_groups,
                                     padding=conv_padding, dropout_prob=dropout_prob, is3d=True)

        # Encoder (downsampling): MaxPool → f_maps[0] → Conv+ReLU → f_maps[1]
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(f_maps[0], f_maps[1], encoder=True,
                       order=layer_order, num_groups=num_groups,
                       padding=conv_padding, dropout_prob=dropout_prob, is3d=True)
        )

        # Decoder (upsampling): Upsample → concat → f_maps[1] → Conv+ReLU → f_maps[0]
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_conv = DoubleConv(f_maps[0] + f_maps[1], f_maps[0], encoder=False,
                                       order=layer_order, num_groups=num_groups,
                                       padding=conv_padding, dropout_prob=dropout_prob, is3d=True)

        # Output convolution: f_maps[0] → Conv+ReLU → output
        self.output_conv = DoubleConv(f_maps[0], f_maps[0], encoder=False,
                                      order=layer_order, num_groups=num_groups,
                                      padding=conv_padding, dropout_prob=dropout_prob, is3d=True)

        # Final 1x1 convolution
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x, return_logits=False):
        # Input → Conv+ReLU → f_maps[0]
        x1 = self.input_conv(x)

        # MaxPool → f_maps[0] → Conv+ReLU → f_maps[1]
        x2 = self.encoder(x1)

        # Upsampling
        x_up = self.upsampling(x2)

        # Ensure upsampled tensor matches skip connection size
        if x_up.shape[2:] != x1.shape[2:]:
            x_up = torch.nn.functional.interpolate(
                x_up, size=x1.shape[2:], mode='nearest'
            )

        # Concatenate skip connection: concat(x1, x_up)
        x_concat = torch.cat([x1, x_up], dim=1)

        # f_maps[1] → Conv+ReLU → f_maps[0]
        x_dec = self.decoder_conv(x_concat)

        # f_maps[0] → Conv+ReLU → f_maps[0] → output
        x_out = self.output_conv(x_dec)

        # Final convolution
        logits = self.final_conv(x_out)

        if self.final_activation is not None:
            output = self.final_activation(logits)
            if return_logits:
                return output, logits
            return output

        if return_logits:
            return logits, logits
        return logits


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               conv_upscale=conv_upscale,
                                               upsample=upsample,
                                               dropout_prob=dropout_prob,
                                               is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=False)


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(ResidualUNet2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob,
                                             is3d=False)


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)


def is_model_2d(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)
