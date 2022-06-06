import glob, os, torch, sys, inspect
###########################
#add parent dir to sys.path
###########################
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
from lib.utils import setup_seed, natural_key
from tqdm import tqdm
from lib.benchmark_utils import ransac_pose_estimation_correspondences, write_est_trajectory, get_inlier_ratio, get_scene_split
from lib.benchmark import benchmark as benchmark_func
import numpy as np
from lib.utils import AverageMeter 
setup_seed(0)


def run_benchmark(feats_scores, n_points, exp_dir, benchmark, ransac_type='correspondence', ransac_with_mutual=False, inlier_ratio_thres=0.05):
    '''
    Test model on benchmark 3DMatch or 3DLoMatch
    :param feats_scores:
    :param n_points:
    :param exp_dir:
    :param benchmark:
    :param ransac_type: in [correspondence, feature], decides which ransac function we use
    :param rasac_with_mutual: decides whether we use mutual correspondences
    :param inlier_ratio_thres: threshold that decides inlier matching
    :return:
    '''
    gt_folder = f'E:/wenxian/code/Pytorch/Coarse-to-fine-correspondences/configs/benchmarks/{benchmark}'

    exp_dir = os.path.join(exp_dir, benchmark, str(n_points))  # 'coarse'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    inliers_eva = os.path.join(exp_dir, 'inliers_eva')
    if not os.path.exists(inliers_eva):
        os.makedirs(inliers_eva)

    results = dict()
    results['w_mutual'] = {'inlier_ratios':[], 'distances':[]}
    results['wo_mutual'] = {'inlier_ratios':[], 'distances':[]}
    tsfm_est = []
    correspondences_mean = 0
    inlier_ratio = AverageMeter()
    inlier_ratio_list = []
    inlier_ratio_dict = {}
    for i in range(20):
        inlier_ratio_dict[i] = []
    x = 0

    for eachfile in tqdm(feats_scores):
        #########################################
        # 1. take input point clouds
        data = torch.load(eachfile)

        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        src_candidate_id, tgt_candidate_id = data['src_candidate_id'], data['tgt_candidate_id']
        correspondences = data['correspondences']
        confidence = data['confidence'].squeeze(1)

        correspondences_raw_src = src_candidate_id[correspondences[:, 0]].unsqueeze(-1)
        correspondences_raw_tgt = tgt_candidate_id[correspondences[:, 1]].unsqueeze(-1)
        correspondences = torch.cat([correspondences_raw_src, correspondences_raw_tgt], dim=-1)

        rot, trans = data['rot'], data['trans']   # gt
        #########################################
        # 2. do sampling guided by score
        if correspondences.shape[0] == 0:
            cur_item = torch.zeros(size=(1, 2), dtype=torch.int32)
            cur_item[0, 0], cur_item[0, 1] = 0, 0
            correspondences = torch.cat([correspondences, cur_item], dim=0)
        
        correspondences_mean += correspondences.shape[0]
        prob = confidence / confidence.sum()
        if correspondences.shape[0] > n_points:
            idx = np.arange(correspondences.shape[0])
            idx = np.random.choice(idx, size=n_points, replace=False, p=prob.numpy())
            correspondences = correspondences[torch.from_numpy(idx).long()]

        src_candidate = src_pcd[correspondences[:, 0], :]
        tgt_candidate = tgt_pcd[correspondences[:, 1], :]
        correspondences_s = torch.arange(end=src_candidate.shape[0]).unsqueeze(-1).repeat(1, 2)
        ##########################################
        # 3. run ransac
        tsfm = ransac_pose_estimation_correspondences(src_candidate, tgt_candidate, correspondences_s, ransac_with_mutual)
        tsfm_est.append(tsfm)

        ##########################################
        # 4. calculate inlier ratios
        src_node = src_candidate[correspondences_s[:, 0]]
        tgt_node = tgt_candidate[correspondences_s[:, 1]]
        if src_node.shape[0] == 0:
            cur_inlier_ratio = 0.
        else:
            cur_inlier_ratio, inlier_idx, inlier_num = get_inlier_ratio(src_node, tgt_node, rot, trans, inlier_distance_threshold=0.1)
        inlier_ratio.update(cur_inlier_ratio, 1)
        inlier_ratio_list.append(cur_inlier_ratio)

        # 保存inliers的corr索引：src索引，tgt索引 以及内点数
        filename = f'{inliers_eva}/{x}.txt'
        with open(filename, 'w') as f:
            for idx in range(correspondences.shape[0]):
                f.write('\t'.join(map(str, correspondences[idx].numpy().tolist())) +'\t')
                f.write(str(inlier_idx[idx].item()) + '\n')

        filename = f'{exp_dir}/inliers_num.txt'
        with open(filename, 'a') as f:    # 追加写入！！！！！！！！
            f.write(f'Corr_num: {correspondences.shape[0]:4d} ')
            f.write(f'inlier_num: {inlier_num.item():4d} ')
            f.write(f'inlier_ratio: {cur_inlier_ratio.item():.4f} \n')
            x += 1

    tsfm_est = np.array(tsfm_est)
    #print("Minimum number of Correspondences: {}".format(num_corr_min))
    #print("Maximum number of Correspondences: {}".format(num_corr_max))
    correspondences_mean /= len(tsfm_est)
    print("Mean correspondence numbers: {}".format(correspondences_mean))
    print("Inlier Ratio: {}".format(inlier_ratio.avg))
    ###################################################
    # write the estimated
    write_est_trajectory(gt_folder, exp_dir, tsfm_est)       # 写入est.txt

    ###################################################
    # evaluate the results, here only FMR now
    inlier_ratio_list = np.array(inlier_ratio_list)
    benchmark_func(exp_dir, gt_folder)            # result.txt
    split = get_scene_split(benchmark)  # gt地址
    
    inliers = []
    fmrs = []
    inlier_ratio_thres = 0.05
    for ele in split:
        c_inliers = inlier_ratio_list[ele[0]:ele[1]]
        inliers.append(np.mean(c_inliers))
        fmrs.append((np.array(c_inliers) > inlier_ratio_thres).mean())
    with open(os.path.join(exp_dir, 'result'), 'a') as f:
        f.write(f'Inlier ratio: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
        f.write(f'Feature match recall: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')

    print('run finished')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default='E:/wenxian/code/Pytorch/Coarse-to-fine-correspondences/snapshot/tdmatch_enc_dec_test_coarse/3DLoMatch', type=str, help='path to precomputed matching scores')
    parser.add_argument('--benchmark', default='3DLoMatch', type=str, help='Either of [3DMatch, 3DLoMatch]')
    parser.add_argument('--n_points', default=250, type=int, help='number of points used by RANSAC')
    parser.add_argument('--exp_dir', default='E:/wenxian/code/Pytorch/Coarse-to-fine-correspondences/est_traj_coarse', type=str, help='export final results')
    args = parser.parse_args()

    feats_scores = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)
    #print(feats_scores)
    run_benchmark(feats_scores, args.n_points, args.exp_dir, args.benchmark)


