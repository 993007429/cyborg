import torch
from torch import nn
import torchvision

from .resnet import resnet50
from .fpn import FPN
from .position_encoding import build_position_encoding


class ResNet34_BackBone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34_BackBone, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5, conv4, conv3, conv2


class ResNet50_BackBone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_BackBone, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5, conv4, conv3, conv2


class ResNet101_BackBone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101_BackBone, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5, conv4, conv3, conv2


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, position_embedding, transformer):
        super(Backbone, self).__init__()
        self.backbone = resnet50(replace_stride_with_dilation=[False, False, False],
                                 pretrained=False,
                                 norm_layer=FrozenBatchNorm2d)

        self.neck = FPN(in_channels=[64, 256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=6,
                        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'))

        self.position_embedding = position_embedding
        self.transformer = transformer

    def forward(self, images):
        xs = list(self.backbone(images))
        mask = torch.zeros_like(xs[-1][:, 0], dtype=torch.bool)
        pos = self.position_embedding(mask)
        out = self.neck(xs, self.transformer, mask, pos)
        return out


def build_backbone():
    position_embedding = build_position_encoding()
    # transformer = build_transformer(args)
    transformer = None
    backbone = Backbone(position_embedding, transformer)
    return backbone
