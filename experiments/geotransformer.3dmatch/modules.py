import math
from posixpath import expanduser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.utils.point_cloud_utils import pairwise_distance, apply_transform
from geotransformer.utils.torch_utils import index_select
from geotransformer.modules.attention.rpe_attention import GeometricTransformer
from geotransformer.modules.attention.positional_embedding import SinusoidalPositionalEmbedding
from geotransformer.modules.registration.modules import WeightedProcrustes


class GeometricTransformerModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_head, blocks, sigma_d, sigma_a, angle_k):
        super(GeometricTransformerModule, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180. / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.rde_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rae_proj = nn.Linear(hidden_dim, hidden_dim)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = GeometricTransformer(blocks, hidden_dim, num_head, dropout=None, activation_fn='relu')
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def get_geometric_structure_embeddings(self, points):
        with torch.no_grad():
            batch_size, num_point, _ = points.shape

            dist_map = torch.sqrt(pairwise_distance(points, points, clamp=True))  # (B, N, N)
            rde_indices = dist_map / self.sigma_d

            knn_indices = dist_map.topk(k=self.angle_k + 1, dim=2, largest=False)[1]  # (B, N, k)
            knn_indices = knn_indices[:, :, 1:]
            knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, self.angle_k, 3)  # (B, N, k, 3)
            expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
            knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
            ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
            anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
            ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, self.angle_k, 3)  # (B, N, N, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, self.angle_k, 3)  # (B, N, N, k, 3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
            cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
            angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
            rae_indices = angles * self.factor_a

        rde = self.embedding(rde_indices)  # (B, N, N, C)
        rde = self.rde_proj(rde)  # (B, N, N, C)

        rae = self.embedding(rae_indices)  # (B, N, N, k, C)
        rae = self.rae_proj(rae)  # (B, N, N, k, C)
        rae = rae.max(dim=3)[0]  # (B, N, N, C)

        gse = rde + rae  # (B, N, N, C)

        return gse

    def forward(self, ref_points, src_points, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""
        Conditional Transformer with Relative Distance Embedding.

        :param ref_points: torch.Tensor (B, N, 3)
        :param src_points: torch.Tensor (B, M, 3)
        :param ref_feats: torch.Tensor (B, N, C)
        :param src_feats: torch.Tensor (B, M, C)
        :param ref_masks: torch.BoolTensor (B, N) (default: None)
        :param src_masks: torch.BoolTensor (B, M) (default: None)
        :return ref_feats: torch.Tensor (B, N, C)
        :return src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.get_geometric_structure_embeddings(ref_points)
        src_embeddings = self.get_geometric_structure_embeddings(src_points)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        ref_feats, src_feats = self.transformer(
            ref_feats, src_feats, ref_embeddings, src_embeddings, masks0=ref_masks, masks1=src_masks
        )
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        return ref_feats, src_feats


class SuperpointTargetGenerator(nn.Module):
    def __init__(self, num_corr, overlap_thresh=0.1):
        super(SuperpointTargetGenerator, self).__init__()
        self.num_corr = num_corr
        self.overlap_thresh = overlap_thresh

    def forward(self, gt_corr_indices, gt_corr_overlaps):
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_thresh)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]
        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        if gt_corr_indices.shape[0] > self.num_corr:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_corr, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_ref_corr_indices = index_select(gt_ref_corr_indices, sel_indices, dim=0)
            gt_src_corr_indices = index_select(gt_src_corr_indices, sel_indices, dim=0)
            gt_corr_overlaps = index_select(gt_corr_overlaps, sel_indices, dim=0)

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps


class SuperpointMatching(nn.Module):
    def __init__(
            self,
            num_proposal,
            dual_normalization=True
    ):
        super(SuperpointMatching, self).__init__()
        self.num_proposal = num_proposal
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks, src_masks):
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = index_select(ref_feats, ref_indices, dim=0)
        src_feats = index_select(src_feats, src_indices, dim=0)
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=self.num_proposal, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original superpoint indices
        ref_corr_indices = index_select(ref_indices, ref_sel_indices, dim=0)
        src_corr_indices = index_select(src_indices, src_sel_indices, dim=0)

        return ref_corr_indices, src_corr_indices, corr_scores


class PointMatching(nn.Module):
    def __init__(
            self,
            k=3,
            threshold=0.05,
            matching_radius=0.1,
            min_num_corr=3,
            max_num_corr=1500,
            num_registration_iter=5
    ):
        super(PointMatching, self).__init__()
        self.k = k
        self.threshold = threshold
        self.matching_radius = matching_radius
        self.min_num_corr = min_num_corr
        self.max_num_corr = max_num_corr
        self.num_registration_iter = num_registration_iter
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_score_map_and_corr_map(
            self,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map
    ):
        matching_score_map = torch.exp(matching_score_map)
        corr_mask_map = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        num_proposal, ref_length, src_length = matching_score_map.shape
        proposal_indices = torch.arange(num_proposal).cuda()

        ref_topk_scores, ref_topk_indices = matching_score_map.topk(k=self.k, dim=2)  # (B, N, K)
        ref_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, ref_length, self.k)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(num_proposal, ref_length, self.k)
        ref_score_map = torch.zeros_like(matching_score_map)
        ref_score_map[ref_proposal_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_map = torch.logical_and(torch.gt(ref_score_map, self.threshold), corr_mask_map)

        src_topk_scores, src_topk_indices = matching_score_map.topk(k=self.k, dim=1)  # (B, K, N)
        src_proposal_indices = proposal_indices.view(num_proposal, 1, 1).expand(num_proposal, self.k, src_length)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(num_proposal, self.k, src_length)
        src_score_map = torch.zeros_like(matching_score_map)
        src_score_map[src_proposal_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_map = torch.logical_and(torch.gt(src_score_map, self.threshold), corr_mask_map)
        score_map = (ref_score_map + src_score_map) / 2

        corr_map = torch.logical_and(ref_corr_map, src_corr_map)

        return score_map, corr_map

    def compute_transform(
            self,
            ref_knn_points,
            src_knn_points,
            score_map,
            corr_map
    ):
        proposal_indices, ref_indices, src_indices = torch.nonzero(corr_map, as_tuple=True)
        all_ref_corr_points = ref_knn_points[proposal_indices, ref_indices]
        all_src_corr_points = src_knn_points[proposal_indices, src_indices]
        all_corr_scores = score_map[proposal_indices, ref_indices, src_indices]

        if all_corr_scores.shape[0] > self.max_num_corr:
            corr_scores, sel_indices = all_corr_scores.topk(k=self.max_num_corr, largest=True)
            ref_corr_points = index_select(all_ref_corr_points, sel_indices, dim=0)
            src_corr_points = index_select(all_src_corr_points, sel_indices, dim=0)
        else:
            ref_corr_points = all_ref_corr_points
            src_corr_points = all_src_corr_points
            corr_scores = all_corr_scores

        # torch.nonzero is row-major, so the correspondences from the same proposal are consecutive.
        # find the first occurrence of each proposal index, then the chunk of this proposal can be obtained.
        unique_masks = torch.ne(proposal_indices[1:], proposal_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [proposal_indices.shape[0]]
        chunks = [(x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.min_num_corr]
        num_proposal = len(chunks)
        if num_proposal > 0:
            indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
            stacked_ref_corr_points = index_select(all_ref_corr_points, indices, dim=0)  # (total, 3)
            stacked_src_corr_points = index_select(all_src_corr_points, indices, dim=0)  # (total, 3)
            stacked_corr_scores = index_select(all_corr_scores, indices, dim=0)  # (total,)

            max_corr = np.max([y - x for x, y in chunks])
            target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
            indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
            indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total, 3)
            indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (total, 3)

            local_ref_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_ref_corr_points.index_put_([indices0, indices1], stacked_ref_corr_points)
            local_ref_corr_points = local_ref_corr_points.view(num_proposal, max_corr, 3)
            local_src_corr_points = torch.zeros(num_proposal * max_corr, 3).cuda()
            local_src_corr_points.index_put_([indices0, indices1], stacked_src_corr_points)
            local_src_corr_points = local_src_corr_points.view(num_proposal, max_corr, 3)
            local_corr_scores = torch.zeros(num_proposal * max_corr).cuda()
            local_corr_scores.index_put_([indices], stacked_corr_scores)
            local_corr_scores = local_corr_scores.view(num_proposal, max_corr)

            estimated_transforms = self.procrustes(local_src_corr_points, local_ref_corr_points, local_corr_scores)
            all_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), estimated_transforms)
            all_corr_distances = torch.sum((ref_corr_points.unsqueeze(0) - all_aligned_src_corr_points) ** 2, dim=2)
            all_inlier_masks = torch.lt(all_corr_distances, self.matching_radius ** 2)  # (P, N)
            best_index = all_inlier_masks.sum(dim=1).argmax()
            inlier_masks = all_inlier_masks[best_index].float()
            estimated_transform = estimated_transforms[best_index]
        else:
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
            corr_distances = torch.sum((ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
            inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2).float()

        cur_corr_scores = corr_scores * inlier_masks
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_registration_iter - 1):
            aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
            corr_distances = torch.sum((ref_corr_points - aligned_src_corr_points) ** 2, dim=1)
            inlier_masks = torch.lt(corr_distances, self.matching_radius ** 2)
            cur_corr_scores = corr_scores * inlier_masks.float()
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return all_ref_corr_points, all_src_corr_points, all_corr_scores, estimated_transform

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            matching_score_map
    ):
        r"""
        :param ref_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param src_knn_points: torch.Tensor (num_proposal, num_point, 3)
        :param ref_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param src_knn_masks: torch.BoolTensor (num_proposal, num_point)
        :param matching_score_map: torch.Tensor (num_proposal, num_point, num_point)
        :param node_corr_scores: torch.Tensor (num_proposal)

        :return ref_corr_indices: torch.LongTensor (self.num_corr,)
        :return src_corr_indices: torch.LongTensor (self.num_corr,)
        :return corr_scores: torch.Tensor (self.num_corr,)
        """
        score_map, corr_map = self.compute_score_map_and_corr_map(
            ref_knn_masks, src_knn_masks, matching_score_map
        )

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.compute_transform(
            ref_knn_points, src_knn_points, score_map, corr_map
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform