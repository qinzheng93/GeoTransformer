import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(logits, labels, return_precision_and_recall=False):
    # generate weights
    weights = torch.ones_like(labels)
    negative_weights = labels.sum() / labels.shape[0]
    positive_weights = 1 - negative_weights
    weights[labels >= 0.5] = positive_weights
    weights[labels < 0.5] = negative_weights

    # weighted bce loss
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    loss = torch.mean(weights * losses)

    # precision and recall
    if return_precision_and_recall:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = torch.gt(probs, 0.5).float()
            result = preds * labels
            precision = result.sum() / (preds.sum() + 1e-12)
            recall = result.sum() / (labels.sum() + 1e-12)
        return loss, precision, recall
    else:
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, return_precision_and_recall=False):
        super(WeightedBCELoss, self).__init__()
        self.return_precision_and_recall = return_precision_and_recall

    def forward(self, logits, labels):
        return weighted_bce_loss(logits, labels, return_precision_and_recall=self.return_precision_and_recall)
