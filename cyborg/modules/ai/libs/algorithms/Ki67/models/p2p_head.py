from torch import nn
import torch
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, feature_size):
        super(RegressionModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, num_classes, feature_size):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.double_conv(x)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_level, row, line):
        super(AnchorPoints, self).__init__()

        self.stride = 2 ** pyramid_level
        self.row = row
        self.line = line

    def forward(self, image_shape):
        anchor_points = self.generate_anchor_points(self.stride, row=self.row, line=self.line)
        anchor_points = np.insert(anchor_points, anchor_points.shape[0] // 2, np.zeros((1, 2)), axis=0)
        all_anchor_points = self.shift(image_shape, self.stride, anchor_points)
        return torch.from_numpy(all_anchor_points).float().cuda()

    @staticmethod
    def generate_anchor_points(stride, row, line):
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        return anchor_points

    @staticmethod
    def shift(shape, stride, anchor_points):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]
        K = shifts.shape[0]
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((1, K * A, 2))
        return all_anchor_points

class P2PNet(nn.Module):
    def __init__(self, backbone, num_classes=1, pyramid_level = 5, row=2, line=2):
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
        super(P2PNet, self).__init__()
        self.num_classes = num_classes
        self.backbone =  backbone
        self.anchor_points = AnchorPoints(pyramid_level, row, line)
        num_anchor = int(row * line) + 1
        self.regression_branch = RegressionModel(num_features_in=512, num_anchor_points=num_anchor,
                                                 feature_size=256)
        self.classification_branch = ClassificationModel(num_features_in=512, num_anchor_points=num_anchor,
                                                         num_classes=num_classes, feature_size=256)
    def forward(self, x):
        x = self.backbone(x)[0]
        batch_size = x.shape[0]
        anchor_points = self.anchor_points(x.shape[2:]).repeat(batch_size, 1, 1)
        deltas = self.regression_branch(x)
        pred_logits = self.classification_branch(x)
        anchor_points = anchor_points.cuda(deltas.device)
        coords = deltas + anchor_points
        return {'pred_points': coords, 'pred_logits': pred_logits}
