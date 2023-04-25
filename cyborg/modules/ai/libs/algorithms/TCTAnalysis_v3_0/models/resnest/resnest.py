##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from .resnet import ResNet, Bottleneck
import math
import os
import torch.nn as nn
__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

# def resnest50(pretrained=True, root='~/.encoding/models', **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3],
#                    radix=2, groups=1, bottleneck_width=64,
#                    deep_stem=True, stem_width=32, avg_down=True,
#                    avd=True, avd_first=False, **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.hub.load_state_dict_from_url(
#             resnest_model_urls['resnest50'], progress=True))
#     return model

pretrained_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_model')

def resnest50(class_num=2, in_channels=1, pretrained=False, root='~/.encoding/models', **kwargs):

    stem_width = 32

    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=stem_width, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        state_dict = torch.load(os.path.join(pretrained_model_dir, 'resnest50-528c19ca.pth'))
        model.load_state_dict(state_dict)
        # print("Loaded")

    if in_channels != 3:
        model.conv1[0] = nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
    if class_num != 1000:
        model.fc = nn.Linear(512*Bottleneck.expansion, class_num)

    return model


def resnest101(class_num=2, in_channels=1, pretrained=False, root='~/.encoding/models', **kwargs):
    stem_width = 64
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=stem_width, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        state_dict = torch.load(os.path.join(pretrained_model_dir, 'resnest101-22405ba7.pth'))
        model.load_state_dict(state_dict)
        # print("Loaded")

    if in_channels != 3:
        model.conv1[0] = nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
    if class_num != 1000:
        model.fc = nn.Linear(512*Bottleneck.expansion, class_num)

    return model

def resnest200(class_num=2, in_channels=1, pretrained=False, root='~/.encoding/models', **kwargs):
    stem_width = 64
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=stem_width, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        state_dict = torch.load(os.path.join(pretrained_model_dir, 'resnest200-75117900.pth'))
        model.load_state_dict(state_dict)
        # print("Loaded")

    if in_channels != 3:
        model.conv1[0] = nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
    if class_num != 1000:
        model.fc = nn.Linear(512*Bottleneck.expansion, class_num)

    return model

def resnest269(class_num=2, in_channels=1, pretrained=False, root='~/.encoding/models', **kwargs):
    stem_width = 64

    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        state_dict = torch.load(os.path.join(pretrained_model_dir, 'resnest269-0cc87c48.pth'))
        model.load_state_dict(state_dict)
        # print("Loaded")

    if in_channels != 3:
        model.conv1[0] = nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
    if class_num != 1000:
        model.fc = nn.Linear(512*Bottleneck.expansion, class_num)
    return model
