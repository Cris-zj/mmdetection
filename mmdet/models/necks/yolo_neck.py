import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import xavier_init, constant_init
from ..registry import NECKS
from ..utils import ConvModule


def make_neck_layer(inplanes,
                    planes,
                    num_blocks,
                    conv_cfg,
                    norm_cfg,
                    act_cfg):
    layers = []
    fuse_conv = ConvModule(
        inplanes,
        planes,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg
    )
    layers.append(fuse_conv)

    inplanes = planes
    planes *= 2
    for i in range(0, num_blocks):
        layers.append(
            ConvModule(
                inplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        layers.append(
            ConvModule(
                planes,
                inplanes,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    return nn.Sequential(*layers)


@NECKS.register_module
class YOLONeck(nn.Module):
    """YOLO Neck
    YOLO convs for the output from backbone or other necks,
    and then send to YOLO heads to get the final predictions.

    Args:
        num_levels (int): number of levels.
        in_channels (Iterable): number of channels in the input feature map.
        out_channels (Iterable): number of channels in the output feature map.
        num_block (Iterable): number of blocks.
        extra_convs_on_inputs (bool): with extra convs on inputs or not.
            when YOLO with pan neck, it is True.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        act_cfg (dict): dictionary to construct
            and config activation layer.
    """

    def __init__(self,
                 num_levels=3,
                 in_channels=[384, 768, 1024],
                 out_channels=[128, 256, 512],
                 num_blocks=[2, 2, 2],
                 extra_convs_on_inputs=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1,
                                     inplace=True),
                 ):
        super(YOLONeck, self).__init__()

        self.num_levels = num_levels
        assert num_levels == len(in_channels)
        assert num_levels == len(out_channels)
        assert num_levels == len(num_blocks)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.yolo_lateral_convs = []
        self.yolo_extra_convs = []
        self.yolo_convs = []

        for i in range(num_levels):
            if i < num_levels - 1:
                lateral_conv = ConvModule(
                    out_channels[i + 1],
                    out_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                name = 'yolo_lateral_conv{}'.format(i + 1)
                self.add_module(name, lateral_conv)
                self.yolo_lateral_convs.append(name)

                if self.extra_convs_on_inputs:
                    extra_conv = ConvModule(
                        out_channels[i + 1],
                        out_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                    name = 'yolo_extra_conv{}'.format(i + 1)
                    self.add_module(name, extra_conv)
                    self.yolo_extra_convs.append(name)

            yolo_conv = make_neck_layer(
                inplanes=in_channels[i],
                planes=out_channels[i],
                num_blocks=num_blocks[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            name = 'yolo_conv{}'.format(i + 1)
            self.add_module(name, yolo_conv)
            self.yolo_convs.append(name)

        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x):
        x = list(x)
        outs = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            layer = getattr(self, self.yolo_convs[i])
            out = layer(x[i])
            outs[i] = out
            if i > 0:
                lateral_layer = getattr(self, self.yolo_lateral_convs[i - 1])
                out = self.upsample(lateral_layer(out))
                if self.extra_convs_on_inputs:
                    extra_layer = getattr(self, self.yolo_extra_convs[i - 1])
                    x[i - 1] = extra_layer(x[i - 1])
                    x[i - 1] = torch.cat((x[i - 1], out), dim=1)
                else:
                    x[i - 1] = torch.cat((out, x[i - 1]), dim=1)

        return tuple(outs)


@NECKS.register_module
class PANYOLO(YOLONeck):
    """Path Aggregate Network

    Refer to the paper for more details: https://arxiv.org/abs/1803.01534.
    """

    def __init__(self, *args, **kwargs):
        super(PANYOLO, self).__init__(*args, **kwargs)

        # build path-aggregate layers
        self.pan_lateral_convs = []
        self.pan_convs = []

        for i in range(1, self.num_levels):

            lateral_conv = ConvModule(
                self.out_channels[i - 1],
                self.out_channels[i],
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            name = 'pan_lateral_conv{}'.format(i + 1)
            self.add_module(name, lateral_conv)
            self.pan_lateral_convs.append(name)

            yolo_conv = make_neck_layer(
                self.out_channels[i] * 2,
                self.out_channels[i],
                num_blocks=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            name = 'pan_conv{}'.format(i + 1)
            self.add_module(name, yolo_conv)
            self.pan_convs.append(name)

    def init_weights(self):
        super(PANYOLO, self).init_weights()

    def forward(self, x):
        x = list(x)
        outs = [None] * self.num_levels
        for i in range(self.num_levels - 1, -1, -1):
            layer = getattr(self, self.yolo_convs[i])
            out = layer(x[i])
            outs[i] = out
            if i > 0:
                lateral_layer = getattr(self, self.yolo_lateral_convs[i - 1])
                out = self.upsample(lateral_layer(out))
                if self.extra_convs_on_inputs:
                    extra_layer = getattr(self, self.yolo_extra_convs[i - 1])
                    x[i - 1] = extra_layer(x[i - 1])
                    x[i - 1] = torch.cat((x[i - 1], out), dim=1)
                else:
                    x[i - 1] = torch.cat((out, x[i - 1]), dim=1)

        for i in range(1, self.num_levels):
            pan_lateral_layer = getattr(self, self.pan_lateral_convs[i - 1])
            outs[i] = torch.cat((pan_lateral_layer(outs[i - 1]), outs[i]), 1)
            pan_layer = getattr(self, self.pan_convs[i - 1])
            outs[i] = pan_layer(outs[i])
        return tuple(outs)
