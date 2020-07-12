import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import xavier_init, constant_init
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class SPP(nn.Module):
    """Spatial Pyramid Pooling
    Refer to the paper for more details: https://arxiv.org/pdf/1406.4729.

    Args:
        in_channels (int): number of channels in the input feature map.
        out_channels (Iterable): number of channels in the output feature map.
        kernel_size (Iterable): kernel sizes of each conv.
        stride (Iterable): strides of each conv.
        padding (Iterable): paddings of each conv.
        out_pool_size (Iterable): expected output size of max pooling layer
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        activation_cfg (dict): dictionary to construct
            and config activation layer.
    returns:
        a tensor vector with shape [1 x n] is the concentration of
        multi-level pooling
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 out_pool_size,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation_cfg=dict(type='LeakyReLU', negative_slope=0.1,
                                     inplace=True)):
        super(SPP, self).__init__()
        self.inplanes = in_channels
        self.layers = []
        for i in range(len(out_channels)):
            outplanes = out_channels[i]
            layer = ConvModule(
                in_channels=self.inplanes,
                out_channels=outplanes,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation_cfg=activation_cfg
            )
            self.inplanes = outplanes
            layer_name = 'spp_conv{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

        self.out_pool_size = out_pool_size

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x):
        outs = []

        x1, x2, x3 = x

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x3 = layer(x3)

        for i in range(len(self.out_pool_size)-1, -1, -1):
            max_pool = nn.MaxPool2d(self.out_pool_size[i], 1,
                                    self.out_pool_size[i] // 2)
            pooled_x = max_pool(x3)
            outs.append(pooled_x)
        outs.append(x3)
        return x1, x2, torch.cat(outs, dim=1)
