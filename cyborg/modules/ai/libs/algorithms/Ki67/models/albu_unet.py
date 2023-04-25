from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """
    def __init__(self, input_channels=1, num_classes=1, num_filters=32, pretrained=True, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super(AlbuNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.register_buffer('device_id', torch.IntTensor(1))


        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        if self.input_channels != 3:
            n = self.input_channels
            self.conv1[0] = nn.Conv2d(n,64,kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4


        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)


        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x, mask=None):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))
        # print('can I see the size?', center.shape, conv5.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1))


        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        out = self.final(dec0)

        # import pdb; pdb.set_trace()

        # if self.num_classes > 1:
        #     x_out = F.log_softmax(self.final(dec0), dim=1)
        # else:
        #     x_out = self.final(dec0)

        # if self.training:
        #     assert mask is not None, "invalid mask for training mode"
        # if self.num_classes > 1:
        #     criterion = BEC_Jaccard_Loss_softmax()
        # else:
        #     criterion = BEC_Jaccard_Loss
        # self._loss = criterion(out, mask)
        # self._loss  = self.compute_multi_loss(out, mask)

        return out

    @property
    def loss(self):
        return self._loss


    def compute_multi_loss(self, outputs, targets):
        criterion = BEC_Jaccard_Loss()
        loss = 0
        for i in range(self.num_classes):
            loss+=criterion(outputs=outputs[:,i,:,:], targets=targets[:,i,:,:])
        loss/=self.num_classes
        return loss

class BEC_Jaccard_Loss_softmax:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.nll_loss = nn.CrossEntropyLoss()
    def __call__(self, outputs, targets):
        eps = 1e-15
        # jaccard_target = (targets>=0.1).float()
        jaccard_target = targets
        jaccard_output = torch.softmax(outputs, dim=1)
        # import pdb; pdb.set_trace()
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        # import pdb; pdb.set_trace()
        loss = self.alpha * self.nll_loss(outputs, targets.max(1)[1])
        loss -= (1-self.alpha) * torch.log((intersection+eps)/(union-intersection+eps))
        return loss

class BEC_Jaccard_Loss:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.nll_loss = nn.BCEWithLogitsLoss()
    def __call__(self, outputs, targets):
        eps = 1e-15
        # jaccard_target = (targets>=0.1).float()
        jaccard_target = (targets).float()
        jaccard_output = torch.sigmoid(outputs)
        # import pdb; pdb.set_trace()
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        loss = self.alpha * self.nll_loss(outputs, targets)
        loss -= (1-self.alpha) * torch.log((intersection+eps)/(union-intersection+eps))
        return loss