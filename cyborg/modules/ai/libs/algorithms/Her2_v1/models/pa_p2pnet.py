# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import torch
from torch import nn

import numpy as np
import torch.nn.functional as F
from .backbone import build_backbone


class AnchorPoints(nn.Module):
    def __init__(self, row=2, col=2, grid_scale=(32, 32)):
        #grid_scale = (16, 16)
        super(AnchorPoints, self).__init__()

        x_space = grid_scale[0] / row
        y_space = grid_scale[1] / col
        self.deltas = np.array(
            [
                [-x_space, -y_space],
                [x_space, -y_space],
                [0, 0],
                [-x_space, y_space],
                [x_space, y_space]
            ]
        ) / 2

        # self.deltas = np.array(
        #     [
        #         [0, 0],
        #     ]
        # )

        self.grid_scale = np.array(grid_scale)

    def forward(self, images):
        bs, _, h, w = images.shape
        centers = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.grid_scale[0])) + 0.5,
                np.arange(np.ceil(h / self.grid_scale[1])) + 0.5),
            -1) * self.grid_scale

        anchors = np.expand_dims(centers, 2) + self.deltas
        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.flatten(0, 2).repeat(bs, 1, 1)


class PAP2PNet(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, hidden_dim, num_classes, row, col):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            num_classes: number of object classes
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(row, col)
        self.hidden_dim = hidden_dim
        self.num_stage = 1
        self.num_levels = self.backbone.neck.num_outs
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1)

        self.loc_aggr = MLP(hidden_dim * self.num_levels, hidden_dim, 2, self.num_levels)
        self.cls_aggr = MLP(hidden_dim * self.num_levels, hidden_dim, 2, self.num_levels)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.constant_(self.deform_layer.layers[-1].weight.data, 0)
        nn.init.constant_(self.deform_layer.layers[-1].bias.data, 0)

    def forward(self, images):
        features, points = self.backbone(images), self.get_aps(images)

        loc_results, cls_results = [], []
        for stage in range(self.num_stage):
            reg_features, cls_features, points = self.extract_features(features, points, stage=stage)
            points = self.reg_head(reg_features) + points
            logits = self.cls_head(cls_features)
            loc_results.append(points)
            cls_results.append(logits)

        outputs = {'pnt_coords': torch.cat(loc_results, 1), 'cls_logits': torch.cat(cls_results, 1)}
        return outputs

    def extract_features(self,
                         features: list,
                         points: torch.Tensor,
                         align: bool = True,
                         stage: int = 0):
        bs, num_points = points.shape[:2]
        roi_features = torch.zeros(self.num_levels, bs, num_points, self.hidden_dim).cuda(points.device)

        # deformable proposal points
        if stage == 0:
            h, w = features[0].shape[2:]
            #scale = torch.tensor([w, h], dtype=torch.float, device=points.device)
            scale = torch.as_tensor([w,h],device=points.device).float()
            grid = (2.0 * points / self.strides[0] / scale - 1.0).unsqueeze(2)  # for coordinate alignment
            #pre_roi_features = F.grid_sample(features[0], grid, align_corners=align).permute(0, 2, 1, 3).squeeze(-1)
            pre_roi_features = F.grid_sample(features[0], grid, align_corners=align).squeeze(-1).permute(0, 2, 1)
            offsets = self.deform_layer(pre_roi_features).view(bs, -1, 2)
            points = points + offsets

        for i, stride in enumerate(self.strides):
            h, w = features[i].shape[2:]
            #scale = torch.tensor([w, h], dtype=torch.float, device=points.device)
            scale = torch.as_tensor([w,h],device=points.device).float()
            grid = (2.0 * points / stride / scale - 1.0).unsqueeze(2)
            #roi_features[i] = F.grid_sample(features[i], grid, align_corners=align).permute(0, 2, 1, 3).squeeze(-1)
            roi_features[i] = F.grid_sample(features[i], grid, align_corners=align).squeeze(-1).permute(0, 2, 1)

        roi_features = roi_features.permute(1, 2, 0, 3)
        attn_features = roi_features.flatten(2)

        reg_attn = self.loc_aggr(attn_features).softmax(-1).unsqueeze(-1)
        reg_features = (reg_attn * roi_features).sum(dim=2)

        cls_attn = self.cls_aggr(attn_features).softmax(-1).unsqueeze(-1)
        cls_features = (cls_attn * roi_features).sum(dim=2)

        return reg_features, cls_features, points


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, p=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(p))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_model(args):
    backbone = build_backbone(args)

    model = PAP2PNet(
        backbone,
        row=args.row,
        col=args.col,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    )
    return model
