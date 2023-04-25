import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_utils import inconv, down, up, outconv

class UNetPap(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(UNetPap, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        self.in_channels = in_channels
        
        self.inc = inconv(in_channels, 32)  # 80
        self.down1 = down(32, 64)           # 40
        self.down2 = down(64, 96)           # 20
        self.down3 = down(96, 96)           # 10
        self.down4 = down(96, 96)           # 5
        
        self.up1   = up(96, 96, side_ch = 96)
        self.up2   = up(96, 96, side_ch = 96)
        self.up3   = up(96, 64, side_ch = 64)
        self.up4   = up(64, 32)
        self.out   = outconv(32, n_classes)

    def forward(self, x, label=None):
        # be aware that label is 1/2 of the size of x in each spatial dimension
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x,refer_shape=x1.size()[2::])
        
        x = self.out(x)

        if self.training:
            assert label is not None, "invalid label for training mode"
            self._loss  =  F.smooth_l1_loss(x, label)
        return x

    @property
    def loss(self):
        return self._loss
