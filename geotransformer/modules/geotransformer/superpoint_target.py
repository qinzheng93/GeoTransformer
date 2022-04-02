import numpy as np
import torch
import torch.nn as nn


class SuperPointTargetGenerator(nn.Module):
    def __init__(self, num_targets, overlap_threshold):
        super(SuperPointTargetGenerator, self).__init__()
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold

    @torch.no_grad()
    def forward(self, gt_corr_indices, gt_corr_overlaps):
        r"""Generate ground truth superpoint (patch) correspondences.

        Randomly select "num_targets" correspondences whose overlap is above "overlap_threshold".

        Args:
            gt_corr_indices (LongTensor): ground truth superpoint correspondences (N, 2)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)

        Returns:
            gt_ref_corr_indices (LongTensor): selected superpoints in reference point cloud.
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_threshold)
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]

        if gt_corr_indices.shape[0] > self.num_targets:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_targets, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_corr_indices = gt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps
