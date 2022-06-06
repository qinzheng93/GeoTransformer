import os, sys, inspect
###########################
#add parent dir to sys.path
###########################
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import open3d as o3d
import numpy as np
import torch, glob
from dataset.common import to_o3d_pcd
from lib.benchmark import read_trajectory, read_trajectory_info, write_trajectory


def get_blue():
    '''
    Get color blue for rendering
    :return:
    '''
    return [0, 0.651, 0.929]


def get_yellow():
    '''
    Get color yellow for rendering
    :return:
    '''
    return [1, 0.706, 0]


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats




def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if (score_mat.ndim == 2):
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, correspondences, mutual=False, distance_threshold=0.05, ransac_n=3):
    '''
    Run RANSAC estimation based on input correspondences
    :param src_pcd:
    :param tgt_pcd:
    :param correspondences:
    :param mutual:
    :param distance_threshold:
    :param ransac_n:
    :return:
    '''

    #ransac_n = correspondences.shape[0]
    

    if mutual:
        raise NotImplementedError
    else:
        #src_pcd = src_pcd.cuda()
        #tgt_pcd = tgt_pcd.cuda()
        #correspondences = correspondences.cuda()
        
        src_pcd = to_o3d_pcd(to_array(src_pcd))
        tgt_pcd = to_o3d_pcd(to_array(tgt_pcd))
        correspondences = o3d.utility.Vector2iVector(to_array(correspondences))
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(src_pcd, tgt_pcd, correspondences, distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],  o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
    return result_ransac.transformation


def write_est_trajectory(gt_folder, exp_dir, tsfm_est):
    """
    Write the estimated trajectories
    """
    scene_names=sorted(os.listdir(gt_folder))
    count=0
    for scene_name in scene_names:
        gt_pairs, gt_traj = read_trajectory(os.path.join(gt_folder,scene_name,'gt.log'))
        est_traj = []
        for i in range(len(gt_pairs)):
            est_traj.append(tsfm_est[count])
            count+=1

        # write the trajectory
        c_directory=os.path.join(exp_dir,scene_name)
        os.makedirs(c_directory,exist_ok=True)
        write_trajectory(np.array(est_traj),gt_pairs,os.path.join(c_directory, 'est.log'))


def get_scene_split(benchmark):
    '''
    Just to check how many valid fragments each scene has
    :param benchmark:
    :return:
    '''

    assert benchmark in ['3DMatch', '3DLoMatch']
    folder = f'configs/benchmarks/{benchmark}/*/gt.log'

    scene_files = sorted(glob.glob(folder))
    split = []
    count = 0
    for eachfile in scene_files:
        gt_pairs, gt_traj = read_trajectory(eachfile)
        split.append([count, count+len(gt_pairs)])
        count += len(gt_pairs)
    return split


def get_inlier_ratio(src_node, tgt_node, rot, trans, inlier_distance_threshold=0.1):
    '''
    Compute inlier ratios based on input torch tensors
    '''
    src_node = (torch.matmul(rot, src_node.T) + trans).T
    dist = torch.norm(src_node - tgt_node, dim=-1)
    inliers = dist < inlier_distance_threshold
    inliers_num = torch.sum(inliers)
    return inliers_num / src_node.shape[0]


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''

    '''
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return:
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2)
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs
