import random, re
import torch
import numpy as np
import time
import open3d

###########################################
# util methods
###########################################
def natural_key(string_):
    '''
    Sort strings by numbers in name
    :param string_:
    :return:
    '''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def setup_seed(seed):
    '''
    fix random seed for deterministic training
    :param seed: seleted seed for deterministic training
    :return: None
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def square_distance(src, tgt, normalize=False):
    '''
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    '''

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def knn(src, tgt, k, normalize=False):
    '''
    Find K-nearest neighbor when ref==tgt and query==src
    Return index of knn, [B, N, k]
    '''
    dist = square_distance(src, tgt, normalize)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return idx


def point2node(nodes, points):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    M, _ = nodes.size()
    N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx


def point2node_correspondences(src_nodes, src_points, tgt_nodes, tgt_points, point_correspondences, device='cpu'):
    '''
    Based on point correspondences & point2node relationships, calculate node correspondences
    :param src_nodes: Nodes of source point cloud
    :param src_points: Points of source point cloud
    :param tgt_nodes: Nodes of target point cloud
    :param tgt_points: Points of target point cloud
    :param point_correspondences: Ground truth point correspondences
    :return: node_corr_mask: Overlap ratios between nodes
             node_corr: Node correspondences sampled for training
    '''
    #####################################
    # calc visible ratio for each node
    src_visible, tgt_visible = point_correspondences[:, 0], point_correspondences[:, 1]

    src_vis, tgt_vis = torch.zeros((src_points.shape[0])).to(device), torch.zeros((tgt_points.shape[0])).to(device)

    src_vis[src_visible] = 1.
    tgt_vis[tgt_visible] = 1.

    src_vis = src_vis.nonzero().squeeze(1)
    tgt_vis = tgt_vis.nonzero().squeeze(1)

    src_vis_num = torch.zeros((src_nodes.shape[0])).to(device)
    src_tot_num = torch.ones((src_nodes.shape[0])).to(device)

    src_idx = point2node(src_nodes, src_points)
    idx, cts = torch.unique(src_idx, return_counts=True)
    src_tot_num[idx] = cts.float()

    src_idx_ = src_idx[src_vis]
    idx_, cts_ = torch.unique(src_idx_, return_counts=True)
    src_vis_num[idx_] = cts_.float()

    src_node_vis = src_vis_num / src_tot_num

    tgt_vis_num = torch.zeros((tgt_nodes.shape[0])).to(device)
    tgt_tot_num = torch.ones((tgt_nodes.shape[0])).to(device)


    tgt_idx = point2node(tgt_nodes, tgt_points)
    idx, cts = torch.unique(tgt_idx, return_counts=True)
    tgt_tot_num[idx] = cts.float()

    tgt_idx_ = tgt_idx[tgt_vis]
    idx_, cts_ = torch.unique(tgt_idx_, return_counts=True)
    tgt_vis_num[idx_] = cts_.float()

    tgt_node_vis = tgt_vis_num / tgt_tot_num

    src_corr = point_correspondences[:, 0]  # [K]
    tgt_corr = point_correspondences[:, 1]  # [K]

    src_node_corr = torch.gather(src_idx, 0, src_corr)
    tgt_node_corr = torch.gather(tgt_idx, 0, tgt_corr)

    index = src_node_corr * tgt_idx.shape[0] + tgt_node_corr

    index, counts = torch.unique(index, return_counts=True)


    src_node_corr = index // tgt_idx.shape[0]
    tgt_node_corr = index % tgt_idx.shape[0]

    node_correspondences = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)

    node_corr_mask = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)
    node_correspondences[src_node_corr, tgt_node_corr] = counts.float()
    node_correspondences = node_correspondences[:-1, :-1]

    node_corr_sum_row = torch.sum(node_correspondences, dim=1, keepdim=True)
    node_corr_sum_col = torch.sum(node_correspondences, dim=0, keepdim=True)

    node_corr_norm_row = (node_correspondences / (node_corr_sum_row + 1e-10)) * src_node_vis.unsqueeze(1).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_norm_col = (node_correspondences / (node_corr_sum_col + 1e-10)) * tgt_node_vis.unsqueeze(0).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_mask[:-1, :-1] = torch.min(node_corr_norm_row, node_corr_norm_col)
    ############################################################
    # Binary masks
    #node_corr_mask[:-1, :-1] = (node_corr_mask[:-1, :-1] > 0.01)
    #node_corr_mask[-1, :-1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=0), min=0.)
    #node_corr_mask[:-1, -1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=1), min=0.)

    #####################################################
    # Soft weighted mask, best Performance
    node_corr_mask[:-1, -1] = 1. - src_node_vis
    node_corr_mask[-1, :-1] = 1. - tgt_node_vis
    #####################################################

    node_corr = node_corr_mask[:-1, :-1].nonzero()
    return node_corr_mask, node_corr


def correspondences_from_score_max(score, mutual=False, supp=False, certainty=None, return_score=False, thres=None):
    '''
    Return estimated rough matching regions from score matrix
    param: score: score matrix, [N, M]
    return: correspondences [K, 2]
    '''
    score = torch.exp(score)

    row_idx = torch.argmax(score[:-1, :], dim=1)
    row_seq = torch.arange(row_idx.shape[0]).cuda()


    col_idx = torch.argmax(score[:, :-1], dim=0)
    col_seq = torch.arange(col_idx.shape[0]).cuda()


    row_map = torch.zeros_like(score).cuda().bool()
    row_map[row_seq, row_idx] = True
    col_map = torch.zeros_like(score).cuda().bool()
    col_map[col_idx, col_seq] = True
    if mutual:
        sel_map = torch.logical_and(row_map, col_map)[:-1, :-1]
    else:
        sel_map = torch.logical_or(row_map, col_map)[:-1, :-1]

    if thres is not None:
        add_map = (score[:-1, :-1] >= thres)
        sel_map = torch.logical_and(sel_map, add_map)



    correspondences = sel_map.nonzero(as_tuple=False)

    if supp and correspondences.shape[0] == 0:
        correspondences = torch.zeros(1, 2).long().cuda()
    if return_score:
        corr_score = score[correspondences[:, 0], correspondences[:, 1]]
        return correspondences, corr_score.view(-1)
    else:
        return correspondences


def correspondences_from_thres(score, thres=0.0, supp=False, return_score=True):
    '''
    Return estimated rough matching regions from score matrix
    param: score: score matrix, [N, M]
    return: correspondences [K, 2]
    '''
    score = torch.exp(score)

    x = torch.arange(score.shape[0] - 1).cuda().unsqueeze(-1)
    x = x.repeat([1, score.shape[1] - 1])

    y = torch.arange(score.shape[1] - 1).cuda().unsqueeze(0)
    y = y.repeat([score.shape[0] - 1, 1])

    mask = score[:-1, :-1] > thres

    x, y = x[mask].unsqueeze(-1), y[mask].unsqueeze(-1)


    correspondences = torch.cat([x, y], dim=-1)

    if supp and correspondences.shape[0] == 0:
        cur_item = torch.zeros(size=(1, 2), dtype=torch.int32).cuda()
        cur_item[0, 0], cur_item[0, 1] = 0, 0
        correspondences = torch.cat([correspondences, cur_item], dim=0)
    if return_score:
        corr_score = score[correspondences[:, 0], correspondences[:, 1]]
        return correspondences, corr_score.view(-1)
    else:
        return correspondences


def get_fine_grained_correspondences(scores, mutual=False, supp=False, certainty=None, node_corr_conf=None, thres=None):
    '''
    '''
    b, n, m = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1

    src_idx_base = 0
    tgt_idx_base = 0

    correspondences = torch.empty(size=(0, 2), dtype=torch.int32).cuda()
    corr_fine_score = torch.empty(size=(0, 1), dtype=torch.float32).cuda()

    for i in range(b):
        score = scores[i, :, :]
        if node_corr_conf is not None:
            correspondence, fine_score = correspondences_from_score_max(score, mutual=mutual, supp=supp, certainty=certainty, return_score=True, thres=thres)
            fine_score = fine_score * node_corr_conf[i]
        else:
            correspondence = correspondences_from_score_max(score, mutual=mutual, supp=supp, certainty=certainty, return_score=False, thres=thres)

        correspondence[:, 0] += src_idx_base
        correspondence[:, 1] += tgt_idx_base

        correspondences = torch.cat([correspondences, correspondence], dim=0)
        if node_corr_conf is not None:
            corr_fine_score = torch.cat([corr_fine_score, fine_score.unsqueeze(-1)], dim=0)

        src_idx_base += n
        tgt_idx_base += m
    if node_corr_conf is not None:
        return correspondences, corr_fine_score
    else:
        return correspondences


def batched_max_selection(scores):
    b, n, m = scores.shape
    corr_map = torch.zeros_like(scores).bool()

    batch_indices = torch.arange(b).unsqueeze(1).expand(b, n).cuda()
    row_indices = torch.arange(n).unsqueeze(0).expand(b, n).cuda()
    col_indices = torch.argmax(scores, dim=2)
    corr_map[batch_indices, row_indices, col_indices] = True

    batch_indices = torch.arange(b).unsqueeze(1).expand(b, m).cuda()
    col_indices = torch.arange(m).unsqueeze(0).expand(b, m).cuda()
    row_indices = torch.argmax(scores, dim=1)
    corr_map[batch_indices, row_indices, col_indices] = True

    corr_map = corr_map[:, :-1, :-1]

    return corr_map


def get_fine_grained_correspondences_gpu(scores, node_corr_conf):
    scores = torch.exp(scores)

    corr_map = batched_max_selection(scores)
    batch_indices, row_indices, col_indices = torch.nonzero(corr_map, as_tuple=True)

    scores = scores[:, :-1, :-1] * node_corr_conf.view(-1, 1, 1)
    fine_corr_conf = scores[batch_indices, row_indices, col_indices]

    _, n, m = scores.shape
    row_indices = row_indices + batch_indices * n
    col_indices = col_indices + batch_indices * m
    correspondences = torch.stack([row_indices, col_indices], dim=1)
    fine_corr_conf = fine_corr_conf.unsqueeze(-1)

    return correspondences, fine_corr_conf


##################################
# utils objects
##################################

class AverageMeter(object):
    '''
    A class computes and stores the average and current values
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.sq_sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(object):
    '''
    A simple timer
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff


class Logger(object):
    '''
    A simple logger
    '''

    def __init__(self, path):
        self.path = path
        self.fw = open(self.path + '/log', 'a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()


def NMSDownSamplerPytorchCPU(object):
    '''
    Downsample base on NMS(Non-maxima-suppression) implemented via Pytorch running on CPU
    '''

    def __init__(self, radius, k):
        self.radius = radius
        self.k = k

    def do_nms(self, pcd, feat, score, k):
        '''
        Run nms algorithm
        :param self:
        :param pcd: point cloud in shape[N, 3]
        :param feat: feature attached to each point, in shape[N, C]
        :param score: saliency score attached to each point, in shape[N]
        :return: pcd_after_nms: point cloud after nms downsample
                 feat_after_nms: feature after nms downsample
        '''
        num_point = pcd.shape[0]
        if num_point > self.k:
            radius2 = self.radius ** 2
            mask = torch.ones(num_point, dtype=torch.bool)
            sorted_score,  sorted_indices = torch.sort(score, descending=True)
            sorted_pcd = pcd[sorted_indices]
            sorted_feat = feat[sorted_indices]

            indices = []
            for i in range(num_point):
                if mask[i]:
                    indices.append(i)
                    if len(indices) == self.k:
                        break
                    if i + 1 < num_point:
                        current_mask = torch.sum((sorted_pcd[i+1:] - sorted_pcd[i]) ** 2, dim=1) < radius2
                        mask[i + 1:] = mask[i + 1:] & ~current_mask

            pcd_after_nms = sorted_pcd[indices]
            feat_after_nms = sorted_feat[indices]
        else:
            pcd_after_nms = pcd
            feat_after_nms = feat
        return pcd_after_nms, feat_after_nms
