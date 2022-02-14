import torch
import torch.nn as nn
import torch.nn.functional as F


def circle_loss(
        pos_scores,
        neg_scores,
        pos_margin,
        neg_margin,
        pos_optimal,
        neg_optimal,
        log_scale,
        pos_scales=None,
        neg_scales=None
):
    pos_weights = torch.maximum(pos_scores - pos_optimal, torch.zeros_like(pos_scores))
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = torch.maximum(neg_optimal - neg_scores, torch.zeros_like(neg_scores))
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos = torch.logsumexp(log_scale * pos_weights * (pos_scores - pos_margin), dim=-1)
    loss_neg = torch.logsumexp(log_scale * neg_weights * (neg_margin - neg_scores), dim=-1)

    loss = F.softplus(loss_pos + loss_neg) / log_scale

    return loss


class CircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_scores, neg_scores, pos_scales=None, neg_scales=None):
        return circle_loss(
            pos_scores, neg_scores, self.pos_margin, self.neg_margin, self.pos_optimal,
            self.neg_optimal, self.log_scale, pos_scales=pos_scales, neg_scales=neg_scales
        )
