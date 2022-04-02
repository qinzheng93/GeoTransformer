import argparse
import os.path as osp
import time
import glob
import sys
import json

import torch
import numpy as np

from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)
from geotransformer.datasets.registration.threedmatch.utils import (
    get_num_fragments,
    get_scene_abbr,
    get_gt_logs_and_infos,
    compute_transform_error,
    write_log_file,
)

from config import make_cfg


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch'], required=True, help='test benchmark')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd'], required=True, help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def eval_one_epoch(args, cfg, logger):
    features_root = osp.join(cfg.feature_dir, args.benchmark)
    benchmark = args.benchmark

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')
    coarse_matching_meter.register_meter('scene_precision')
    coarse_matching_meter.register_meter('scene_PMR>0')
    coarse_matching_meter.register_meter('scene_PMR>=0.1')
    coarse_matching_meter.register_meter('scene_PMR>=0.3')
    coarse_matching_meter.register_meter('scene_PMR>=0.5')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('scene_recall')
    fine_matching_meter.register_meter('scene_inlier_ratio')
    fine_matching_meter.register_meter('scene_overlap')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('mean_rre')
    registration_meter.register_meter('mean_rte')
    registration_meter.register_meter('median_rre')
    registration_meter.register_meter('median_rte')
    registration_meter.register_meter('scene_recall')
    registration_meter.register_meter('scene_rre')
    registration_meter.register_meter('scene_rte')

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_roots = sorted(glob.glob(osp.join(features_root, '*')))
    for scene_root in scene_roots:
        coarse_matching_meter.reset_meter('scene_precision')
        coarse_matching_meter.reset_meter('scene_PMR>0')
        coarse_matching_meter.reset_meter('scene_PMR>=0.1')
        coarse_matching_meter.reset_meter('scene_PMR>=0.3')
        coarse_matching_meter.reset_meter('scene_PMR>=0.5')

        fine_matching_meter.reset_meter('scene_recall')
        fine_matching_meter.reset_meter('scene_inlier_ratio')
        fine_matching_meter.reset_meter('scene_overlap')

        registration_meter.reset_meter('scene_recall')
        registration_meter.reset_meter('scene_rre')
        registration_meter.reset_meter('scene_rte')

        scene_name = osp.basename(scene_root)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)
        gt_root = osp.join(cfg.data.dataset_root, 'metadata', 'benchmarks', benchmark, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_root, num_fragments)

        estimated_transforms = []

        file_names = sorted(
            glob.glob(osp.join(scene_root, '*.npz')),
            key=lambda x: [int(i) for i in osp.basename(x).split('.')[0].split('_')],
        )
        for file_name in file_names:
            ref_frame, src_frame = [int(x) for x in osp.basename(file_name).split('.')[0].split('_')]

            data_dict = np.load(file_name)

            ref_points_c = data_dict['ref_points_c']
            src_points_c = data_dict['src_points_c']
            ref_node_corr_indices = data_dict['ref_node_corr_indices']
            src_node_corr_indices = data_dict['src_node_corr_indices']

            ref_corr_points = data_dict['ref_corr_points']
            src_corr_points = data_dict['src_corr_points']
            corr_scores = data_dict['corr_scores']

            gt_node_corr_indices = data_dict['gt_node_corr_indices']
            transform = data_dict['transform']
            pcd_overlap = data_dict['overlap']

            if args.num_corr is not None and corr_scores.shape[0] > engine.args.num_corr:
                sel_indices = np.argsort(-corr_scores)[: args.num_corr]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]

            message = '{}, id0: {}, id1: {}, OV: {:.3f}'.format(scene_abbr, ref_frame, src_frame, pcd_overlap)

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                ref_points_c, src_points_c, ref_node_corr_indices, src_node_corr_indices, gt_node_corr_indices
            )

            coarse_precision = coarse_matching_result_dict['precision']

            coarse_matching_meter.update('scene_precision', coarse_precision)
            coarse_matching_meter.update('scene_PMR>0', float(coarse_precision > 0))
            coarse_matching_meter.update('scene_PMR>=0.1', float(coarse_precision >= 0.1))
            coarse_matching_meter.update('scene_PMR>=0.3', float(coarse_precision >= 0.3))
            coarse_matching_meter.update('scene_PMR>=0.5', float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                ref_corr_points, src_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict['inlier_ratio']
            overlap = fine_matching_result_dict['overlap']

            fine_matching_meter.update('scene_inlier_ratio', inlier_ratio)
            fine_matching_meter.update('scene_overlap', overlap)
            fine_matching_meter.update('scene_recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

            message += ', c_PIR: {:.3f}'.format(coarse_precision)
            message += ', f_IR: {:.3f}'.format(inlier_ratio)
            message += ', f_OV: {:.3f}'.format(overlap)
            message += ', f_RS: {:.3f}'.format(fine_matching_result_dict['residual'])
            message += ', f_NU: {}'.format(fine_matching_result_dict['num_corr'])

            # 2. evaluate registration
            if args.method == 'lgr':
                estimated_transform = data_dict['estimated_transform']
            elif args.method == 'ransac':
                estimated_transform = registration_with_ransac_from_correspondences(
                    src_corr_points,
                    ref_corr_points,
                    distance_threshold=cfg.ransac.distance_threshold,
                    ransac_n=cfg.ransac.num_points,
                    num_iterations=cfg.ransac.num_iterations,
                )
            elif args.method == 'svd':
                with torch.no_grad():
                    ref_corr_points = torch.from_numpy(ref_corr_points).cuda()
                    src_corr_points = torch.from_numpy(src_corr_points).cuda()
                    corr_scores = torch.from_numpy(corr_scores).cuda()
                    estimated_transform = weighted_procrustes(
                        src_corr_points, ref_corr_points, corr_scores, return_transform=True
                    )
                    estimated_transform = estimated_transform.detach().cpu().numpy()
            else:
                raise ValueError(f'Unsupported registration method: {args.method}.')

            estimated_transforms.append(
                dict(
                    test_pair=[ref_frame, src_frame],
                    num_fragments=num_fragments,
                    transform=estimated_transform,
                )
            )

            if gt_indices[ref_frame, src_frame] != -1:
                # evaluate transform (realignment error)
                gt_index = gt_indices[ref_frame, src_frame]
                transform = gt_logs[gt_index]['transform']
                covariance = gt_infos[gt_index]['covariance']
                error = compute_transform_error(transform, covariance, estimated_transform)
                message += ', r_RMSE: {:.3f}'.format(np.sqrt(error))
                accepted = error < cfg.eval.rmse_threshold ** 2
                registration_meter.update('scene_recall', float(accepted))
                if accepted:
                    rre, rte = compute_registration_error(transform, estimated_transform)
                    registration_meter.update('scene_rre', rre)
                    registration_meter.update('scene_rte', rte)
                    message += ', r_RRE: {:.3f}'.format(rre)
                    message += ', r_RTE: {:.3f}'.format(rte)

            # Evaluate re-alignment error
            # if ref_frame + 1 < src_frame:
            #     evaluate transform (realignment error)
            #     src_points_f = data_dict['src_points_f']
            #     error = compute_realignment_error(src_points_f, transform, estimated_transform)
            #     message += ', r_RMSE: {:.3f}'.format(error)
            #     accepted = error < config.eval_rmse_threshold
            #     registration_meter.update('scene_recall', float(accepted))
            #     if accepted:
            #         rre, rte = compute_registration_error(transform, estimated_transform)
            #         registration_meter.update('scene_rre', rre)
            #         registration_meter.update('scene_rte', rte)
            #         message += ', r_RRE: {:.3f}, r_RTE: {:.3f}'.format(rre, rte)

            if args.verbose:
                logger.info(message)

        est_log = osp.join(cfg.registration_dir, benchmark, scene_name, 'est.log')
        write_log_file(est_log, estimated_transforms)

        logger.info(f'Scene_name: {scene_name}')

        # 1. print correspondence evaluation results (one scene)
        # 1.1 coarse level statistics
        coarse_precision = coarse_matching_meter.mean('scene_precision')
        coarse_matching_recall_0 = coarse_matching_meter.mean('scene_PMR>0')
        coarse_matching_recall_1 = coarse_matching_meter.mean('scene_PMR>=0.1')
        coarse_matching_recall_3 = coarse_matching_meter.mean('scene_PMR>=0.3')
        coarse_matching_recall_5 = coarse_matching_meter.mean('scene_PMR>=0.5')
        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('PMR>0', coarse_matching_recall_0)
        coarse_matching_meter.update('PMR>=0.1', coarse_matching_recall_1)
        coarse_matching_meter.update('PMR>=0.3', coarse_matching_recall_3)
        coarse_matching_meter.update('PMR>=0.5', coarse_matching_recall_5)
        scene_coarse_matching_result_dict[scene_abbr] = {
            'precision': coarse_precision,
            'PMR>0': coarse_matching_recall_0,
            'PMR>=0.1': coarse_matching_recall_1,
            'PMR>=0.3': coarse_matching_recall_3,
            'PMR>=0.5': coarse_matching_recall_5,
        }

        # 1.2 fine level statistics
        recall = fine_matching_meter.mean('scene_recall')
        inlier_ratio = fine_matching_meter.mean('scene_inlier_ratio')
        overlap = fine_matching_meter.mean('scene_overlap')
        fine_matching_meter.update('recall', recall)
        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('overlap', overlap)
        scene_fine_matching_result_dict[scene_abbr] = {'recall': recall, 'inlier_ratio': inlier_ratio}

        message = '  Correspondence, '
        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', c_PMR>0: {:.3f}'.format(coarse_matching_recall_0)
        message += ', c_PMR>=0.1: {:.3f}'.format(coarse_matching_recall_1)
        message += ', c_PMR>=0.3: {:.3f}'.format(coarse_matching_recall_3)
        message += ', c_PMR>=0.5: {:.3f}'.format(coarse_matching_recall_5)
        message += ', f_FMR: {:.3f}'.format(recall)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_OV: {:.3f}'.format(overlap)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall = registration_meter.mean('scene_recall')
        mean_rre = registration_meter.mean('scene_rre')
        mean_rte = registration_meter.mean('scene_rte')
        median_rre = registration_meter.median('scene_rre')
        median_rte = registration_meter.median('scene_rte')
        registration_meter.update('recall', recall)
        registration_meter.update('mean_rre', mean_rre)
        registration_meter.update('mean_rte', mean_rte)
        registration_meter.update('median_rre', median_rre)
        registration_meter.update('median_rte', median_rte)

        scene_registration_result_dict[scene_abbr] = {
            'recall': recall,
            'mean_rre': mean_rre,
            'mean_rte': mean_rte,
            'median_rre': median_rre,
            'median_rte': median_rte,
        }

        message = '  Registration'
        message += ', RR: {:.3f}'.format(recall)
        message += ', mean_RRE: {:.3f}'.format(mean_rre)
        message += ', mean_RTE: {:.3f}'.format(mean_rte)
        message += ', median_RRE: {:.3f}'.format(median_rre)
        message += ', median_RTE: {:.3f}'.format(median_rte)
        logger.info(message)

    if args.test_epoch is not None:
        logger.critical('Epoch {}'.format(args.test_epoch))

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_coarse_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', PIR: {:.3f}'.format(result_dict['precision'])
        message += ', PMR>0: {:.3f}'.format(result_dict['PMR>0'])
        message += ', PMR>=0.1: {:.3f}'.format(result_dict['PMR>=0.1'])
        message += ', PMR>=0.3: {:.3f}'.format(result_dict['PMR>=0.3'])
        message += ', PMR>=0.5: {:.3f}'.format(result_dict['PMR>=0.5'])
        logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.3f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', OV: {:.3f}'.format(fine_matching_meter.mean('overlap'))
    message += ', std: {:.3f}'.format(fine_matching_meter.std('recall'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_fine_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', FMR: {:.3f}'.format(result_dict['recall'])
        message += ', IR: {:.3f}'.format(result_dict['inlier_ratio'])
        logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.3f}'.format(registration_meter.mean('recall'))
    message += ', mean_RRE: {:.3f}'.format(registration_meter.mean('mean_rre'))
    message += ', mean_RTE: {:.3f}'.format(registration_meter.mean('mean_rte'))
    message += ', median_RRE: {:.3f}'.format(registration_meter.mean('median_rre'))
    message += ', median_RTE: {:.3f}'.format(registration_meter.mean('median_rte'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_registration_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', RR: {:.3f}'.format(result_dict['recall'])
        message += ', mean_RRE: {:.3f}'.format(result_dict['mean_rre'])
        message += ', mean_RTE: {:.3f}'.format(result_dict['mean_rte'])
        message += ', median_RRE: {:.3f}'.format(result_dict['median_rre'])
        message += ', median_RTE: {:.3f}'.format(result_dict['median_rte'])
        logger.critical(message)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
