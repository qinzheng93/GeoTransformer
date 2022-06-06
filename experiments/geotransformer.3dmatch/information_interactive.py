import copy
import torch
import torch.nn as nn
from information_process import square_dists, gather_points, sample_and_group, angle


def get_graph_features(feats, coords, k=10):
    '''

    :param feats: (B, N, C)
    :param coords: (B, N, 3)
    :param k: float
    :return: (B, N, k, 2C)
    '''

    sq_dists = square_dists(coords, coords)
    n = coords.size(1)
    inds = torch.topk(sq_dists, min(n, k+1), dim=-1, largest=False, sorted=True)[1]
    inds = inds[:, :, 1:] # (B, N, k)

    neigh_feats = gather_points(feats, inds) # (B, N, k, c)
    feats = torch.unsqueeze(feats, 2).repeat(1, 1, min(n-1, k), 1) # (B, N, k, c)
    return torch.cat([feats, neigh_feats - feats], dim=-1)


class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'in_{i}',
                                   nn.InstanceNorm2d(out_dims))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''
        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x


class PPF(nn.Module):
    def __init__(self, feats_dim, k, radius):
        super().__init__()
        self.k = k
        self.radius = radius
        self.local_feature_fused = LocalFeatureFused(in_dim=10,
                                                     out_dims=feats_dim)
    
    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, 3, N)
        :param k: int
        :return: (B, C, N)
        '''

        feats = feats.permute(0, 2, 1).contiguous()
        coords = coords.permute(0, 2, 1).contiguous()
        new_xyz, new_points, grouped_inds, grouped_xyz = \
            sample_and_group(xyz=coords,
                             points=feats,
                             M=-1,
                             radius=self.radius,
                             K=self.k)
        nr_d = angle(feats[:, :, None, :], grouped_xyz)
        ni_d = angle(new_points[..., 3:], grouped_xyz)
        nr_ni = angle(feats[:, :, None, :], new_points[..., 3:])
        d_norm = torch.norm(grouped_xyz, dim=-1)
        ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1) # (B, N, K, 4)
        new_points = torch.cat([new_points[..., :3], ppf_feat], dim=-1)

        coords = torch.unsqueeze(coords, dim=2).repeat(1, 1, min(self.k, new_points.size(2)), 1)
        new_points = torch.cat([coords, new_points], dim=-1)
        feature_local = new_points.permute(0, 3, 2, 1).contiguous() # (B, C1 + 3, K, N)
        feature_local = self.local_feature_fused(feature_local)
        return feature_local


class GCN(nn.Module):
    def __init__(self, feats_dim, k):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim, 1, bias=False),
            nn.InstanceNorm2d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim * 2, 1, bias=False),
            nn.InstanceNorm2d(feats_dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(feats_dim * 4, feats_dim, 1, bias=False),
            nn.InstanceNorm1d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.k = k

    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, C, N)
        :param k: int
        :return: (B, C, N)
        '''
        feats1 = get_graph_features(feats=feats.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)        # (B, N, K, 2*C)
        feats1 = self.conv1(feats1.permute(0, 3, 1, 2).contiguous())   # (B, C, N, K)
        feats1 = torch.max(feats1, dim=-1)[0] # (B, C, N)

        feats2 = get_graph_features(feats=feats1.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)
        feats2 = self.conv2(feats2.permute(0, 3, 1, 2).contiguous())
        feats2 = torch.max(feats2, dim=-1)[0] # (B, 2*C, N)

        feats3 = torch.cat([feats, feats1, feats2], dim=1) #  (B, 4*C, N)
        feats3 = self.conv3(feats3)  # (B, C, N)

        return feats3


class GGE(nn.Module):
    def __init__(self, feats_dim, gcn_k, ppf_k, radius, bottleneck):
        super().__init__()
        self.gcn = GCN(feats_dim, gcn_k)
        if bottleneck:
            self.ppf = PPF([feats_dim // 2, feats_dim, feats_dim // 2], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim + feats_dim // 2, 1),
                nn.InstanceNorm1d(feats_dim + feats_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
        else:
            self.ppf = PPF([feats_dim, feats_dim*2, feats_dim], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(feats_dim * 2, feats_dim * 2, 1),
                nn.InstanceNorm1d(feats_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim * 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
    
    def forward(self, coords, feats, normals):
        feats_ppf = self.ppf(coords, normals)
        feats_gcn = self.gcn(coords, feats)
        feats_fused = self.fused(torch.cat([feats_ppf, feats_gcn], dim=1))
        return feats_fused   # (B, C, N)


def multi_head_attention(query, key, value):
    '''

    :param query: (B, dim, nhead, N)
    :param key: (B, dim, nhead, M)
    :param value: (B, dim, nhead, M)
    :return: (B, dim, nhead, N)
    '''
    dim = query.size(1)
    scores = torch.einsum('bdhn, bdhm->bhnm', query, key) / dim**0.5
    attention = torch.nn.functional.softmax(scores, dim=-1)
    feats = torch.einsum('bhnm, bdhm->bdhn', attention, value)
    return feats


class Cross_Attention(nn.Module):
    def __init__(self, feat_dims, nhead):
        super().__init__()
        assert feat_dims % nhead == 0
        self.feats_dim = feat_dims
        self.nhead = nhead
        # self.q_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.k_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.v_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        self.conv = nn.Conv1d(feat_dims, feat_dims, 1)
        self.q_conv, self.k_conv, self.v_conv = [copy.deepcopy(self.conv) for _ in range(3)] # a good way than better ?
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dims * 2, feat_dims * 2, 1),
            nn.InstanceNorm1d(feat_dims * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dims * 2, feat_dims, 1),
        )

    def forward(self, feats1, feats2):
        '''

        :param feats1: (B, C, N)
        :param feats2: (B, C, M)
        :return: (B, C, N)
        '''
        b = feats1.size(0)
        dims = self.feats_dim // self.nhead
        query = self.q_conv(feats1).reshape(b, dims, self.nhead, -1)
        key = self.k_conv(feats2).reshape(b, dims, self.nhead, -1)
        value = self.v_conv(feats2).reshape(b, dims, self.nhead, -1)
        feats = multi_head_attention(query, key, value)
        feats = feats.reshape(b, self.feats_dim, -1)
        feats = self.conv(feats)
        cross_feats = self.mlp(torch.cat([feats1, feats], dim=1))
        return cross_feats


class InformationInteractive(nn.Module):
    def __init__(self, layer_names, feat_dims, gcn_k, ppf_k, radius, bottleneck, nhead):
        super().__init__()
        self.layer_names = layer_names
        self.blocks = nn.ModuleList()
        for layer_name in layer_names:
            if layer_name == 'gcn':
                self.blocks.append(GCN(feat_dims, gcn_k))
            elif layer_name == 'gge':
                self.blocks.append(GGE(feat_dims, gcn_k, ppf_k, radius, bottleneck))
            elif layer_name == 'cross_attn':
                self.blocks.append(Cross_Attention(feat_dims, nhead))
            else:
                raise NotImplementedError

    def forward(self, coords1, feats1, coords2, feats2, normals1, normals2):
        '''

        :param coords1: (B, 3, N)
        :param feats1: (B, C, N)
        :param coords2: (B, 3, M)
        :param feats2: (B, C, M)
        :return: feats1=(B, C, N), feats2=(B, C, M)
        '''
        for layer_name, block in zip(self.layer_names, self.blocks):
            if layer_name == 'gcn':
                feats1 = block(coords1, feats1)
                feats2 = block(coords2, feats2)
            elif layer_name == 'gge':
                feats1 = block(coords1, feats1, normals1)
                feats2 = block(coords2, feats2, normals2)
            elif layer_name == 'cross_attn':
                feats1 = feats1 + block(feats1, feats2)
                feats2 = feats2 + block(feats2, feats1)
            else:
                raise NotImplementedError

        return feats1, feats2
