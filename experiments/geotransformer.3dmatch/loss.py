import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.utils.point_cloud_utils import apply_transform, pairwise_distance
from geotransformer.utils.registration_utils import compute_registration_error
from geotransformer.modules.loss.circle_loss import WeightedCircleLoss


class OverlapAwareCircleLoss(nn.Module):
    def __init__(self, config):
        super(OverlapAwareCircleLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            config.overlap_aware_circle_loss_positive_margin,
            config.overlap_aware_circle_loss_negative_margin,
            config.overlap_aware_circle_loss_positive_optimal,
            config.overlap_aware_circle_loss_negative_optimal,
            config.overlap_aware_circle_loss_log_scale
        )
        self.pos_thresh = config.overlap_aware_circle_loss_positive_threshold

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.pos_thresh)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class PointMatchingLoss(nn.Module):
    def __init__(self, config):
        super(PointMatchingLoss, self).__init__()
        self.pos_radius = config.point_matching_loss_positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.pos_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, config):
        super(OverallLoss, self).__init__()
        self.suerpoint_loss = OverlapAwareCircleLoss(config)
        self.point_loss = PointMatchingLoss(config)
        self.weight_superpoint_loss = config.weight_superpoint_loss
        self.weight_point_loss = config.weight_point_loss

    def forward(self, output_dict, data_dict):
        superpoint_loss = self.suerpoint_loss(output_dict)
        point_loss = self.point_loss(output_dict, data_dict)

        loss = self.weight_superpoint_loss * superpoint_loss + self.weight_point_loss * point_loss

        result_dict = {
            'c_loss': superpoint_loss,
            'f_loss': point_loss,
            'loss': loss
        }

        return result_dict


class Evaluator(nn.Module):
    def __init__(self, config):
        super(Evaluator, self).__init__()
        self.positive_overlap = config.superpoint_matching_positive_overlap
        self.positive_radius = config.point_matching_positive_radius

    @torch.no_grad()
    def evaluate_superpoint_matching(self, output_dict, data_dict):
        ref_length_c = data_dict['stack_lengths'][-1][0]
        src_length_c = data_dict['stack_lengths'][-1][1]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.positive_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_point_matching(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.positive_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        rre, rte = compute_registration_error(transform, est_transform)
        return rre, rte

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_superpoint_matching(output_dict, data_dict)
        f_precision = self.evaluate_point_matching(output_dict, data_dict)
        rre, rte = self.evaluate_registration(output_dict, data_dict)

        result_dict = {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte
        }
        return result_dict
