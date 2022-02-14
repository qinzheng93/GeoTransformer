import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.kpconv.kpfcnn import make_kpfcnn_encoder, make_kpfcnn_decoder, KPEncoder, KPDecoder
from geotransformer.modules.optimal_transport.modules import LearnableLogOptimalTransport
from geotransformer.utils.point_cloud_utils import get_point_to_node_indices_and_masks
from geotransformer.utils.registration_utils import get_node_corr_indices_and_overlaps

from modules import GeometricTransformerModule, SuperpointMatching, PointMatching, SuperpointTargetGenerator


class Node2Point(nn.Module):
    def __init__(self, config):
        super(Node2Point, self).__init__()
        self.final_feats_dim = config.final_feats_dim
        self.point_to_node_max_point = config.point_to_node_max_point
        self.pos_radius = config.ground_truth_positive_radius

        # KPConv Encoder
        encoder_dict = make_kpfcnn_encoder(config, config.in_features_dim)
        self.encoder = KPEncoder(encoder_dict)

        # GNN part
        self.transformer = GeometricTransformerModule(
            encoder_dict['out_dim'],
            config.geometric_transformer_feats_dim,
            config.geometric_transformer_feats_dim,
            config.geometric_transformer_num_head,
            config.geometric_transformer_architecture,
            config.geometric_transformer_sigma_d,
            config.geometric_transformer_sigma_a,
            config.geometric_transformer_angle_k
        )

        # KPConv Decoder
        decoder_dict = make_kpfcnn_decoder(config, encoder_dict, encoder_dict['out_dim'], config.final_feats_dim)
        self.decoder = KPDecoder(decoder_dict)

        # Optimal Transport
        self.optimal_transport = LearnableLogOptimalTransport(config.sinkhorn_num_iter)

        # Correspondence Generator
        self.superpoint_matching = SuperpointMatching(
            config.superpoint_matching_num_proposal,
            dual_normalization=config.superpoint_matching_dual_normalization
        )

        self.point_matching = PointMatching(
            k=config.point_matching_topk,
            threshold=config.point_matching_confidence_threshold,
            matching_radius=config.point_matching_positive_radius,
            min_num_corr=config.point_matching_min_num_corr,
            max_num_corr=config.point_matching_max_num_corr,
            num_registration_iter=config.point_matching_num_registration_iter
        )

        # Target Generator
        self.superpoint_target_generator = SuperpointTargetGenerator(
            config.superpoint_matching_num_target,
            overlap_thresh=config.superpoint_matching_overlap_thresh
        )

    def forward(self, data_dict):
        output_dict = {}

        feats_f = data_dict['features'].detach()
        ref_length_c = data_dict['stack_lengths'][-1][0].item()
        ref_length_m = data_dict['stack_lengths'][1][0].item()
        ref_length_f = data_dict['stack_lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_m = data_dict['points'][1].detach()
        points_f = data_dict['points'][0].detach()
        transform = data_dict['transform'].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_m = points_m[:ref_length_m]
        src_points_m = points_m[ref_length_m:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_m'] = ref_points_m
        output_dict['src_points_m'] = src_points_m
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = get_point_to_node_indices_and_masks(
            ref_points_m, ref_points_c, self.point_to_node_max_point
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = get_point_to_node_indices_and_masks(
            src_points_m, src_points_c, self.point_to_node_max_point
        )

        sentinel_point = torch.zeros(1, 3).cuda()
        ref_padded_points_m = torch.cat([ref_points_m, sentinel_point], dim=0)
        src_padded_points_m = torch.cat([src_points_m, sentinel_point], dim=0)

        ref_node_knn_points = ref_padded_points_m[ref_node_knn_indices]
        src_node_knn_points = src_padded_points_m[src_node_knn_indices]
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_corr_indices_and_overlaps(
            ref_points_c, src_points_c, ref_node_knn_points, src_node_knn_points, transform, self.pos_radius,
            ref_masks=ref_node_masks, src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks, src_knn_masks=src_node_knn_masks
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_c, skip_feats = self.encoder(feats_f, data_dict)

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0), src_points_c.unsqueeze(0), ref_feats_c.unsqueeze(0), src_feats_c.unsqueeze(0)
        )

        # 4. Head for coarse level matching
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. KPFCNN Decoder
        feats_m = self.decoder(feats_c, skip_feats, data_dict)

        # 5. Head for fine level matching
        ref_feats_m = feats_m[:ref_length_m]
        src_feats_m = feats_m[ref_length_m:]
        output_dict['ref_feats_m'] = ref_feats_m
        output_dict['src_feats_m'] = src_feats_m

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.superpoint_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

        # 7 Random select ground truth node correspondences during training
        if self.training:
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.superpoint_target_generator(
                gt_node_corr_indices, gt_node_corr_overlaps
            )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        sentinel_feat = torch.zeros(1, self.final_feats_dim).cuda()
        ref_padded_feats_m = torch.cat([ref_feats_m, sentinel_feat], dim=0)
        src_padded_feats_m = torch.cat([src_feats_m, sentinel_feat], dim=0)
        ref_node_corr_knn_feats = ref_padded_feats_m[ref_node_corr_knn_indices]  # (P, K, C)
        src_node_corr_knn_feats = src_padded_feats_m[src_node_corr_knn_indices]  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / self.final_feats_dim ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            matching_scores = matching_scores[:, :-1, :-1]
            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.point_matching(
                ref_node_corr_knn_points, src_node_corr_knn_points,
                ref_node_corr_knn_masks, src_node_corr_knn_masks,
                matching_scores
            )
            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config):
    model = Node2Point(config)
    return model


def main():
    from config import config
    model = create_model(config)
    print(model)


if __name__ == '__main__':
    main()
