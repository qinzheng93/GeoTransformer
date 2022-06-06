from lib.trainer import Trainer
import os, torch
from tqdm import tqdm
import numpy as np
from lib.utils import get_fine_grained_correspondences, correspondences_from_score_max
from lib.benchmark_utils import ransac_pose_estimation_correspondences, to_array, get_angle_deviation
# modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler


class TDMatchTester(Trainer):
    '''
    3DMatch Tester
    '''

    def __init__(self, args):
        Trainer.__init__(self, args)

    def test(self):
        print('Start to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        total_corr_num = 0
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                inputs = c_loader_iter.next()
                ##############################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)

                #############################
                # Forward pass
                len_src_pcd = inputs['stack_lengths'][0][0]
                pcds = inputs['points'][0]
                src_pcd, tgt_pcd = pcds[:len_src_pcd], pcds[len_src_pcd:]

                len_src_nodes = inputs['stack_lengths'][-1][0]
                nodes = inputs['points'][-1]
                src_node, tgt_node = nodes[:len_src_nodes], nodes[len_src_nodes:]

                rot = inputs['rot']
                trans = inputs['trans']

                src_candidates_c, tgt_candidates_c, local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel = self.model.forward(inputs)
                
                total_corr_num += node_corr.shape[0]

                correspondences, corr_conf = get_fine_grained_correspondences(local_scores, mutual=False, supp=False, node_corr_conf=node_corr_conf)

                data = dict()
                data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
                data['src_node'], data['tgt_node'] = src_node.cpu(), tgt_node.cpu()
                data['src_candidate'], data['tgt_candidate'] = src_candidates_c.view(-1, 3).cpu(), tgt_candidates_c.view(-1, 3).cpu()
                data['src_candidate_id'], data['tgt_candidate_id'] = src_pcd_sel.cpu(), tgt_pcd_sel.cpu()
                data['rot'] = rot.cpu()
                data['trans'] = trans.cpu()
                data['correspondences'] = correspondences.cpu()
                data['confidence'] = corr_conf.cpu()

                torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')

        print(f'Avg Node Correspondences: {total_corr_num/num_iter}')


class KITTITester(Trainer):
    """
    KITTI tester
    """

    def __init__(self, args):
        Trainer.__init__(self, args)

    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()

        self.model.eval()
        rot_gt, trans_gt = [], []
        with torch.no_grad():
            for _ in tqdm(range(num_iter)):  # loop through this epoch
                inputs = c_loader_iter.next()
                ###############################################
                # forward pass
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                len_src_nodes = inputs['stack_lengths'][-1][0]
                nodes = inputs['points'][-1]

                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())

                src_candidates_c, tgt_candidates_c, local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel = self.model.forward(inputs)
                correspondences, corr_conf = get_fine_grained_correspondences(local_scores, mutual=False, supp=False, node_corr_conf=node_corr_conf)
                ########################################
                # run probabilistic sampling
                n_points = 5000

                if (correspondences.shape[0] > n_points):
                    idx = np.arange(correspondences.shape[0])
                    prob = corr_conf.squeeze(1)
                    prob = prob / prob.sum()
                    idx = np.random.choice(idx, size=n_points, replace=False, p=prob.cpu().numpy())
                    correspondences = correspondences.cpu().numpy()[idx]

                src_pcd_reg = src_candidates_c.view(-1, 3).cpu().numpy()[correspondences[:, 0]]
                tgt_pcd_reg = tgt_candidates_c.view(-1, 3).cpu().numpy()[correspondences[:, 1]]
                correspondences = torch.arange(end=src_pcd_reg.shape[0]).unsqueeze(-1).repeat(1, 2).numpy()
                ########################################
                # run ransac
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation_correspondences(src_pcd_reg, tgt_pcd_reg, correspondences,
                                                                distance_threshold=distance_threshold, ransac_n=4)
                tsfm_est.append(ts_est)

        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:, :3, :3]
        trans_est = tsfm_est[:, :3, 3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:, :, 0]

        rot_threshold = 5
        trans_threshold = 2

        np.savez(f'{self.snapshot_dir}/results', rot_est=rot_est, rot_gt=rot_gt, trans_est=trans_est, trans_gt=trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est - trans_gt, axis=-1)

        flag_1 = r_deviation < rot_threshold
        flag_2 = translation_errors < trans_threshold
        correct = (flag_1 & flag_2).sum()
        precision = correct / rot_gt.shape[0]

        message = f'\n Registration recall: {precision:.3f}\n'

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors = dict()
        errors['rot_mean'] = round(np.mean(r_deviation), 3)
        errors['rot_median'] = round(np.median(r_deviation), 3)
        errors['trans_rmse'] = round(np.mean(translation_errors), 3)
        errors['trans_rmedse'] = round(np.median(translation_errors), 3)
        errors['rot_std'] = round(np.std(r_deviation), 3)
        errors['trans_std'] = round(np.std(translation_errors), 3)

        message += str(errors)
        print(message)
        self.logger.write(message + '\n')


def compute_metrics(data , pred_transforms):
    """
    Compute metrics required in the paper
    """
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_array(t_mse),
            't_mae': to_array(t_mae),
            'err_r_deg': to_array(residual_rotdeg),
            'err_t': to_array(residual_transmag),
            'chamfer_dist': to_array(chamfer_dist)
        }

    return metrics


def print_metrics(logger, summary_metrics , losses_by_iteration=None,title='Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))

def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def get_trainer(config):
    '''
    :param config:
    :return:
    '''

    if config.dataset == 'tdmatch':
        return TDMatchTester(config)
    elif config.dataset == 'kitti':
        return KITTITester(config)
    else:
        raise NotImplementedError
