import torch
import torch.nn as nn

from .functional import weighted_procrustes


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0., eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights):
        return weighted_procrustes(
            src_points, tgt_points, weights, weight_thresh=self.weight_thresh, eps=self.eps,
            return_transform=self.return_transform
        )
