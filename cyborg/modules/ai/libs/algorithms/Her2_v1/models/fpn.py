import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class FPN(nn.Module):
    def __init__(self, in_channels, num_outs, out_channel=256, pretrained=False):
        super(FPN, self).__init__()
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for inc in in_channels:
            lateral_conv = nn.Conv2d(inc, out_channel, kernel_size=(1, 1), stride=(1, 1))
            fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            if not pretrained:
                nn.init.kaiming_normal_(lateral_conv.weight, a=0, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(lateral_conv.bias, 0)

                nn.init.kaiming_normal_(fpn_conv.weight, a=0, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(fpn_conv.bias, 0)

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        if pretrained:
            url = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
            state_dict = load_state_dict_from_url(url, progress=True)
            model_dict = self.state_dict()
            for i in range(4):
                model_dict[f'lateral_convs.{i}.weight'] = state_dict[f'backbone.fpn.inner_blocks.{i}.weight']
                model_dict[f'lateral_convs.{i}.bias'] = state_dict[f'backbone.fpn.inner_blocks.{i}.bias']

                model_dict[f'fpn_convs.{i}.weight'] = state_dict[f'backbone.fpn.layer_blocks.{i}.weight']
                model_dict[f'fpn_convs.{i}.bias'] = state_dict[f'backbone.fpn.layer_blocks.{i}.bias']
            self.load_state_dict(model_dict)

    def forward(self, inputs, transformer=None, mask=None, pos=None):
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if transformer is not None:
            laterals[-1] = transformer(laterals[-1], mask, pos)

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        for i in range(self.num_outs - used_backbone_levels):
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return outs
