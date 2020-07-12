import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer,
                    build_activation_layer


# A help function to build a 'conv-bn-activation' module
def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       conv_cfg=None,
                       norm_cfg=dict(type='BN'),
                       activation_cfg=dict(type='LeakyReLU',
                                           negative_slope=0.1)):
    layers = []
    layers.append(build_conv_layer(conv_cfg,
                                   inplanes,
                                   planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=False))
    layers.append(build_norm_layer(norm_cfg, planes)[1])
    layers.append(build_activation_layer(activation_cfg))
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        """Residual Block for DarkNet.
        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inplanes,
                                                  postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes,
                                                  postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.add_module(self.norm2_name, norm2)

        self.activation = build_activation_layer(activation_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out += identity

        return out


# TODO: Insert the CSPNet to other backbones
class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.

    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concated into fuse transition layer.

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        activation_cfg (dict): dictionary to construct
            and config activation layer.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


def make_dark_layer(block,
                    inplanes,
                    planes,
                    num_blocks,
                    conv_cfg,
                    norm_cfg,
                    activation_cfg):
    downsample = ConvNormActivation(
        inplanes=inplanes,
        planes=planes,
        kernel_size=3,
        stride=2,
        padding=1,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        activation_cfg=activation_cfg
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                downsample=downsample if i == 0 else None,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation_cfg=activation_cfg
            )
        )
    return nn.Sequential(*layers)


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       conv_cfg,
                       norm_cfg,
                       activation_cfg):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        activation_cfg=activation_cfg
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation_cfg=activation_cfg
            )
        )
    return nn.Sequential(*layers)


@BACKBONES.register_module
class DarkNet(nn.Module):
    """DarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767

    Args:
        depth (int): Depth of Darknet, from {21, 53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        activation_cfg (dict): dictionary to construct
            and config activation layer.
    """

    arch_settings = {
        53: (DarkBlock, (1, 2, 8, 8, 4))
    }

    def __init__(self,
                 depth,
                 num_stages=5,
                 with_csp=False,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 activation_cfg=dict(type='LeakyReLU',
                                     negative_slope=0.1, inplace=True),
                 with_classifier=False,
                 num_classes=1000):
        super(DarkNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        assert 1 <= num_stages <= 5
        self.num_stages = num_stages
        assert max(out_indices) < num_stages
        self.with_csp = with_csp
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.activation_cfg = activation_cfg

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 32

        self._make_stem_layer()

        self.dark_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2**i
            if not self.with_csp:
                layer = make_dark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation_cfg=self.activation_cfg
                )
            else:
                layer = make_cspdark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    is_csp_first_stage=True if i == 0 else False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation_cfg=self.activation_cfg
                )
                layer = CrossStagePartialBlock(
                    self.inplanes,
                    planes,
                    stage_layers=layer,
                    is_csp_first_stage=True if i == 0 else False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation_cfg=self.activation_cfg
                )
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.dark_layers.append(layer_name)

        self._freeze_stages()

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = build_norm_layer(self.norm_cfg, self.inplanes)[1]
        self.act1 = build_activation_layer(self.activation_cfg)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        frozen_stages = self.frozen_stages \
            if self.frozen_stages <= len(self.stage_blocks) \
            else len(self.stage_blocks)

        for i in range(1, frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from mmdet.apis import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        outs = []

        for i, layer_name in enumerate(self.dark_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(DarkNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
