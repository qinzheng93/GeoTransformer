"""
Script for benchmarking the 3DMatch test dataset.

Author: Zan Gojcic, Shengyu Huang
Last modified: 30.11.2020
"""

import numpy as np
import os, sys, glob, torch, math
from collections import defaultdict
import nibabel.quaternions as nq

def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]

    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]

    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]

    Returns:
        te (torch tensor): translation error in meters [b,1]

    """
    return torch.norm(t1-t2, dim=(1, 2))

def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html
    
    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]

    Returns:
    p (float): transformation error
    """
    
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]
    
    return p.item()

def read_trajectory(filename, dim=4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim, dim] 
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim+1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])


        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split('\t')[0:dim])

        traj = np.asarray(traj, dtype=np.float).reshape(-1,dim,dim)
        
        final_keys = np.asarray(final_keys)

        return final_keys, traj


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim] 
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)
    
    cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1,dim,dim)
    
    return n_frame, cov_matrix

def extract_corresponding_trajectors(est_pairs,gt_pairs, gt_traj):
    """
    Extract only those transformation matrices from the ground truth trajectory that are also in the estimated trajectory.
    
    Args:
    est_pairs (numpy array): indices of point cloud pairs with enough estimated overlap [m, 3]
    gt_pairs (numpy array): indices of gt overlaping point cloud pairs [n,3]
    gt_traj (numpy array): 3d array of the gt transformation parameters [n,4,4]

    Returns:
    ext_traj (numpy array): gt transformation parameters for the point cloud pairs from est_pairs [m,4,4] 
    """
    ext_traj = np.zeros((len(est_pairs), 4, 4))

    for est_idx, pair in enumerate(est_pairs):
        pair[2] = gt_pairs[0][2]
        gt_idx = np.where((gt_pairs == pair).all(axis=1))[0]
        
        ext_traj[est_idx,:,:] = gt_traj[gt_idx,:,:]

    return ext_traj

def write_trajectory(traj,metadata, filename, dim=4):
    """
    Writes the trajectory into a '.txt' file in 3DMatch/Redwood format. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    traj (numpy array): trajectory for n pairs[n,dim, dim] 
    metadata (numpy array): file containing metadata about fragment numbers [n,3]
    filename (str): path where to save the '.txt' file containing trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    """

    with open(filename, 'w') as f:
        for idx in range(traj.shape[0]):
            # Only save the transfromation parameters for which the overlap threshold was satisfied
            if metadata[idx][2]:
                p = traj[idx,:,:].tolist()
                f.write('\t'.join(map(str, metadata[idx])) + '\n')
                f.write('\n'.join('\t'.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
                f.write('\n')


def read_pairs(src_path,tgt_path,n_points):
    # get pointcloud
    src = torch.load(src_path)
    tgt = torch.load(tgt_path)
    src_pcd, src_embedding = src['coords'],src['feats']
    tgt_pcd, tgt_embedding = tgt['coords'], tgt['feats']
    
    #permute and randomly select 2048/1024 points
    if(src_pcd.shape[0]>n_points):
        src_permute=np.random.permutation(src_pcd.shape[0])[:n_points]
    else:
        src_permute=np.random.choice(src_pcd.shape[0],n_points)
    if(tgt_pcd.shape[0]>n_points):
        tgt_permute=np.random.permutation(tgt_pcd.shape[0])[:n_points]
    else:
        tgt_permute=np.random.choice(tgt_pcd.shape[0],n_points)

    src_pcd,src_embedding = src_pcd[src_permute],src_embedding[src_permute]
    tgt_pcd,tgt_embedding = tgt_pcd[tgt_permute],tgt_embedding[tgt_permute]
    return src_pcd,src_embedding,tgt_pcd,tgt_embedding


def evaluate_registration(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2):
    """
    Evaluates the performance of the registration algorithm according to the evaluation protocol defined
    by the 3DMatch/Redwood datasets. The evaluation protocol can be found at http://redwood-data.org/indoor/registration.html
    
    Args:
    num_fragment (int): path to the '.txt' file containing the trajectory information data
    result (numpy array): estimated transformation matrices [n,4,4]
    result_pairs (numpy array): indices of the point cloud for which the transformation matrix was estimated (m,3)
    gt_pairs (numpy array): indices of the ground truth overlapping point cloud pairs (n,3)
    gt (numpy array): ground truth transformation matrices [n,4,4]
    gt_cov (numpy array): covariance matrix of the ground truth transfromation parameters [n,6,6]
    err2 (float): threshold for the RMSE of the gt correspondences (default: 0.2m)

    Returns:
    precision (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    err2 = err2 ** 2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int)
    flags=[]

    for idx in range(gt_pairs.shape[0]):
        i = int(gt_pairs[idx,0])
        j = int(gt_pairs[idx,1])

        # Only non consecutive pairs are tested
        if j - i > 1:
            gt_mask[i, j] = idx

    n_gt = np.sum(gt_mask > 0)

    good = 0
    n_res = 0
    for idx in range(result_pairs.shape[0]):
        i = int(result_pairs[idx,0])
        j = int(result_pairs[idx,1])
        pose = result[idx,:,:]

        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx,:,:]) @ pose, gt_info[gt_idx,:,:])
            if p <= err2:
                good += 1
                flags.append(0)
            else:
                flags.append(1)
        else:
            flags.append(2)
    if n_res == 0:
        n_res += 1e6
    precision = good * 1.0 / n_res
    recall = good * 1.0 / n_gt

    return precision, recall, flags

def benchmark(est_folder,gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder,ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids= []

    short_names=['Kitchen','Home 1','Home 2','Hotel 1','Hotel 2','Hotel 3','Study','MIT Lab']
    with open(f'{est_folder}/result','w') as f:
        f.write(("Scene\t¦ prec.\t¦ rec.\t¦ re\t¦ te\t¦ samples\t¦\n"))

        for idx,scene in enumerate(scene_names):
            # ground truth info
            gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
            n_valid=0
            for ele in gt_pairs:
                diff=abs(int(ele[0])-int(ele[1]))
                n_valid+=diff>1
            n_valids.append(n_valid)

            n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene,"gt.info"))

            # estimated info
            est_pairs, est_traj = read_trajectory(os.path.join(est_folder,scenes[idx],'est.log'))


            temp_precision, temp_recall,c_flag = evaluate_registration(n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov)
            
            # Filter out the estimated rotation matrices
            ext_gt_traj = extract_corresponding_trajectors(est_pairs,gt_pairs, gt_traj)

            re = rotation_error(torch.from_numpy(ext_gt_traj[:,0:3,0:3]), torch.from_numpy(est_traj[:,0:3,0:3])).cpu().numpy()[np.array(c_flag)==0]
            te = translation_error(torch.from_numpy(ext_gt_traj[:,0:3,3:4]), torch.from_numpy(est_traj[:,0:3,3:4])).cpu().numpy()[np.array(c_flag)==0]


            re_per_scene['mean'].append(np.mean(re))
            re_per_scene['median'].append(np.median(re))
            re_per_scene['min'].append(np.min(re))
            re_per_scene['max'].append(np.max(re))
            

            te_per_scene['mean'].append(np.mean(te))
            te_per_scene['median'].append(np.median(te))
            te_per_scene['min'].append(np.min(te))
            te_per_scene['max'].append(np.max(te))


            re_all.extend(re.reshape(-1).tolist())
            te_all.extend(te.reshape(-1).tolist())

            precision.append(temp_precision)
            recall.append(temp_recall)

            f.write("{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:3d}¦\n".format(short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid))
            np.save(f'{est_folder}/{scenes[idx]}/flag.npy',c_flag)
        
        weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

        f.write("Mean precision: {:.3f}: +- {:.3f}\n".format(np.mean(precision),np.std(precision)))
        f.write("Mean Recall: {:.3f}: +- {:.3f}\n".format(np.mean(recall),np.std(recall)))
        f.write("Weighted precision: {:.3f}\n".format(weighted_precision))

        f.write("Mean median RRE: {:.3f}: +- {:.3f}\n".format(np.mean(re_per_scene['median']), np.std(re_per_scene['median'])))
        f.write("Mean median RTE: {:.3F}: +- {:.3f}\n".format(np.mean(te_per_scene['median']),np.std(te_per_scene['median'])))
    f.close()
    