import torch
import torch.nn as nn


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iterations, inf=1e12):
        r"""Sinkhorn Optimal transport with dustbin parameter (SuperGlue style)."""
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iterations = num_iterations
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.0)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks=None, col_masks=None):
        r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores: torch.Tensor (B, M, N)
            row_masks: torch.Tensor (B, M)
            col_masks: torch.Tensor (B, N)

        Returns:
            matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        if row_masks is None:
            row_masks = torch.ones(size=(batch_size, num_row), dtype=torch.bool).cuda()
        if col_masks is None:
            col_masks = torch.ones(size=(batch_size, num_col), dtype=torch.bool).cuda()

        padded_row_masks = torch.zeros(size=(batch_size, num_row + 1), dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(size=(batch_size, num_col + 1), dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores.masked_fill_(padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iterations={})'.format(self.num_iterations)
        return format_string
