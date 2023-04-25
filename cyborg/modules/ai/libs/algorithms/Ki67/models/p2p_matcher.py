# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb
import numpy as np
# from model.utils import get_world_size, is_dist_avail_and_initialized

class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = ['labels', 'points']
        self.weight_dict = {'loss_ce': args.loss_cls, 'loss_points': args.loss_points}

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = args.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points.float(), target_points.float(), reduction='none')

        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def loss_labels(self, outputs, targets, indices, num_points):
        # print('src_logits_size', src_logits.size())
        # indices: list, b1 [[(x1, x2, x3, ..., xn), (y1, y2, y3, ..., yn)],
        #                b2  [(x1, x2, x3, ..., xm), (y1, y2, y3, ..., ym)],
        #                b3  [(x1, x2, x3, ..., xs), (y1, y2, y3, ..., ys)],
        #                b4  [(x1, x2, x3, ..., xt), (y1, y2, y3, ..., yt)]]
        # targets: list [[targets['point'], targets['labels']],
        #                [targets['point'], targets['labels']],
        #                [targets['point'], targets['labels']],
        #                [targets['point'], targets['labels']], ]
        # targets['point']: list[[x1, y1], [x2, y2], ..., [xn, yn]]
        # targets['labels']: list[[label1, label2, ..., labeln]]
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # indices shape: list, [(source_num, target_num), (source_num, target_num), (source_num, target_num), ...,]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # batch_idx shape: list, fill with batch_id, [(source_num, ) cat (source_num, ) cat (source_num, ), ..., ]
        src_idx = torch.cat([src for (src, _) in indices])
        # src_idx shape: list, source id[(source_num, ) cat (source_num, ) cat (source_num, ), ..., ]
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points, min=1).item()
        #
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        weight_dict = self.weight_dict
        losses = [losses[k] * weight_dict[k] for k in losses.keys()]
        # pdb.set_trace()

        return losses, indices1



class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ids = torch.cat([v["labels"].cuda(out_prob.device) for v in targets])
        tgt_points = torch.cat([v["points"].cuda(out_prob.device) for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L2 cost between point
        # print('out and target shape', out_points.size(), tgt_points.size())
        cost_point = torch.cdist(out_points.float(), tgt_points.float(), p=2)

        # Compute the giou cost between point

        # Final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # print('-----------------------------------------')
        # print(indices[0], '\n',indices[1], '\n',indices[2], '\n',indices[3])
        # print('-----------------------------------------')
        # print(len(indices[0][0]), '\n', len(indices[1][0]), '\n', len(indices[2][0]), '\n', len(indices[3][0]))
        # print(len(indices[0][1]), '\n', len(indices[1][1]), '\n', len(indices[2][1]), '\n', len(indices[3][1]))
        # print(len(indices[4][1]), '\n', len(indices[1][1]), '\n', len(indices[2][1]), '\n', len(indices[3][1]))
        # print('-----------------------------------------')
        indices_results = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices_results

def get_match_items(indices, outputs, targets, label_dict, device_id):
    src_logits = outputs['pred_logits']
    src_points = outputs['pred_points']
    num_classes = src_logits.size()[-1]

    ## get all_pred num
    argmax_label = torch.argmax(src_logits, dim=-1)
    all_pred_num_per_class_list = []
    for i in range(num_classes):
        num_per_class = (argmax_label == i).sum()
        if i > 0:
            all_pred_num_per_class_list.append(num_per_class.detach().cpu())
        print('all_pred_num_{}: '.format(label_dict[str(i)]), num_per_class)

    ## get target num
    all_target = torch.cat([t['labels'] for t in targets])
    all_target_num_per_class_list = []
    for i in range(1, num_classes):
        num_per_class = (all_target == i).sum()
        all_target_num_per_class_list.append(num_per_class.detach().cpu())
        print('target_num_{}: '.format(label_dict[str(i)]), num_per_class)

    ## get detection match pred num
    src_scores = torch.cat([output[output_id] for output, (output_id, _) in zip(src_logits, indices)])
    device = torch.device("cuda:{}".format(device_id))
    batch_id = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]).to(device)
    pred_match_points = torch.cat([points[output_id] for points, (output_id, _) in zip(src_points, indices)]).to(device)
    target_match_points = torch.cat([t['points'][target_id] for t, (_, target_id) in zip(targets, indices)]).to(device)
    target_match_labels = torch.cat([t['labels'][target_id] for t, (_, target_id) in zip(targets, indices)]).to(device)

    _, pred_match_labels = torch.max(src_scores, dim=1)
    match_prob = torch.gather(src_scores, 1, target_match_labels[:, None])
    label_match_bool = (pred_match_labels == target_match_labels)
    label_match_bool = torch.squeeze(label_match_bool)

    dist_threshold = 16
    dist = ((pred_match_points[:, 0] - target_match_points[:, 0]) ** 2 +
            (pred_match_points[:, 1] - target_match_points[:, 1]) ** 2) ** 0.5
    dist_bool = (dist < dist_threshold)
    dist_bool = torch.squeeze(dist_bool)
    match_bool = label_match_bool * dist_bool
    match_bool = match_bool.to(device)

    if match_bool.sum() > 0:
        print('Yeah! I got points')
        batch_id = torch.masked_select(batch_id, match_bool)
        match_points = torch.masked_select(pred_match_points, match_bool.unsqueeze(-1)).reshape(-1, 2)
        match_labels = torch.masked_select(pred_match_labels, match_bool)
        match_prob = torch.masked_select(match_prob, match_bool)

        match_cls_num_per_class_list = []
        for i in range(1, num_classes):
            num_match_per_class = (match_labels == i).sum()
            match_cls_num_per_class_list.append(num_match_per_class.detach().cpu())
            print('match_cls_num_{}: '.format(label_dict[str(i)]), num_match_per_class)
    else:
        batch_id = None
        match_points =None
        match_labels = None
        match_cls_num_per_class_list = [0, 0, 0, 0, 0, 0]

    stat_dict = {'pred_num': np.array(all_pred_num_per_class_list),
                 'target_num': np.array(all_target_num_per_class_list),
                 'match_cls_num': np.array(match_cls_num_per_class_list)}

    return batch_id, match_points, match_labels, match_prob, stat_dict

# def vis_match_results(batch_id, match_labels, match_points, target_points, width = 256, height = 256):
#     batch_id = np.array(batch_id)
#     match_labels = np.array(match_labels)
#     match_points = np.array(match_points)
#     target_points = np.array(target_points)
#     color_dict = {"0": (0, 255, 0), "1": (255, 0, 0), "2": (255, 255, 0), "3": (139, 105, 20),
#                   "4": (0, 0, 255), "5": (128, 0, 255)}
#     radius = 5
#     img = np.uint8(np.zeros((4, width, height, 3)))
#     for batch_idx, match_label, match_point, target_point in zip(batch_id, match_labels, match_points, target_points):
#
#         img[batch_idx] = cv2.circle(img[batch_idx], (match_point[0], match_point[1]), radius, color_dict[str(match_label)])
#         img[batch_idx] = cv2.circle(img[batch_idx], (target_point[0], target_point[1]), radius, color_dict[str(match_label)], -1)
#
#     imsave('/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/save_images/img_1.jpg', img[0])
#     imsave('/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/save_images/img_2.jpg', img[1])
#     imsave('/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/save_images/img_3.jpg', img[2])
#     imsave('/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/save_images/img_4.jpg', img[3])
#     return

# if __name__ == '__main__':
#
#     weight_dict = {'loss_ce': 1, 'loss_points': 1}
#     num_classes = 6
#     batch_size = 4
#     num_pred_points = 256
#     output_class = torch.nn.functional.softmax(torch.rand(batch_size, num_pred_points, num_classes), dim=-1)
#     width, height = 256, 256
#     threshold = 0.2
#     yv, xv = torch.meshgrid([torch.arange(0, height, 16), torch.arange(0, width, 16)])
#     coords_per_sample = torch.stack((xv, yv), 2).view((-1, 2))
#     output_coord = torch.stack([torch.cat([coords_per_sample]) for i in range(batch_size)], dim=0)
#     print(output_coord.size())
#     outputs = {'pred_logits': output_class, 'pred_points': output_coord}
#
#
#     targets = []
#     for batch_id in range(batch_size):
#         num_target_points = torch.randint(200, (1, ))[0]
#         print('num_target_points', num_target_points)
#         target_points = torch.randint(256, (num_target_points, 2))
#         target_labels = torch.randint(num_classes, (num_target_points,))
#         target_dict_per_sample = {'point': target_points, 'labels': target_labels}
#         targets.append(target_dict_per_sample)
#
#
#     losses = ['labels', 'points']
#     matcher = HungarianMatcher_Crowd(cost_class = 1, cost_point = 0.05)
#     criterion = SetCriterion_Crowd(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses)
#
#     indices, loss_dict = criterion(outputs, targets)
#     weight_dict = criterion.weight_dict
#     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#
#     matched_pred_num, all_pred_num, all_target_num = \
#         get_match_items(indices, outputs, targets, threshold = threshold)
#     true_positive = matched_pred_num
#     False_positive = all_pred_num - matched_pred_num
#     False_negative = all_target_num - matched_pred_num
#     eps = 1e-5
#     Precision = true_positive / (true_positive + False_positive + eps)
#     Recall = true_positive / (true_positive + False_negative + eps)
#     F1_score = 2 * Precision * Recall / (Precision + Recall + eps)
#
#     print('Precision', Precision, '\n',
#           'Recall', Recall, '\n',
#           'F1_score', F1_score, '\n')



    # matched_pred, parsed_target = get_final_reuslts(match_labels, match_points)
    # matched_pred_num, target_num = len(matched_pred), len(parsed_target)
    # save_match_results(matched_pred, parsed_target)

    # print('matched_pred_num', 'target_num', matched_pred_num, target_num)
    # print('------------------match_labels--------------------------')
    # print(match_labels, match_points.size())
    # print('------------------match_points--------------------------')
    # print(match_points, match_points.size())
    # print('---------------------------------------------------------')



    # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    # for t, point_scores in zip(targets, outputs_scores):
    #     print(t['labels'])
    #     point_scores_watch = point_scores[:, t['labels']]
    #     print('point_scores', point_scores_watch)
    #     print('point_scores_shape', point_scores_watch.size())
    #     # print(batch_id, t, point_scores)
    #     pdb.set_trace()
    # print('outputs_scores', outputs_scores)
    # target_scores = torch.cat([point_scores[:, t['labels']] for t, point_scores in zip(targets, outputs_scores)], dim = 1)
    # print('target_scores_size', target_scores.size())
    #
    # threshold = 0.2
    # matched_pred_num = int((target_scores > threshold).sum())
    # all_output_scores = target_scores.size()[0] * target_scores.size()[1]
    # unmatched_pred_num = all_output_scores - matched_pred_num
    # print('results: ', matched_pred_num, unmatched_pred_num)

    # losses = ['labels', 'points']
    # matcher = HungarianMatcher_Crowd(cost_class = 1, cost_point = 0.05)
    # criterion = SetCriterion_Crowd(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses)
    # indices1, losses = criterion(outputs, targets)
    # print('--------------------------------------------')
    # print(losses)
