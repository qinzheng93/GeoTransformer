import argparse
import os
import os.path as osp
import time
import glob

import numpy as np
from tqdm import tqdm
from IPython import embed

from geotransformer.utils.metrics import StatisticsDictMeter
from geotransformer.engine import Engine
from geotransformer.utils.registration_utils import evaluate_correspondences
from geotransformer.datasets.registration.threedmatch_helpers import (
    write_log_file, evaluate_registration_3dmatch, get_num_fragment, get_scene_abbr
)

from config import config


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--run_matching', action='store_true', help='whether evaluate correspondence')
    parser.add_argument('--run_registration', action='store_true', help='whether evaluate registration')
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'extra'], help='test benchmark')
    return parser


def eval_superpoint_matching(ref_points, src_points, ref_corr_indices, src_corr_indices, correspondences):
    ref_gt_corr_indices = correspondences[:, 0]
    src_gt_corr_indices = correspondences[:, 1]

    gt_node_corr_map = np.zeros((ref_points.shape[0], src_points.shape[0]))
    gt_node_corr_map[ref_gt_corr_indices, src_gt_corr_indices] = 1.
    num_gt_node_corr = gt_node_corr_map.sum() + 1e-12

    pred_node_corr_map = np.zeros_like(gt_node_corr_map)
    pred_node_corr_map[ref_corr_indices, src_corr_indices] = 1.
    num_pred_node_corr = pred_node_corr_map.sum() + 1e-12

    pos_node_corr_map = gt_node_corr_map * pred_node_corr_map
    num_pos_node_corr = pos_node_corr_map.sum()

    precision = num_pos_node_corr / num_pred_node_corr
    recall = num_pos_node_corr / num_gt_node_corr

    return precision, recall


def eval_one_epoch(engine):
    features_root = osp.join(config.features_dir, engine.args.benchmark)

    benchmark = engine.args.benchmark
    run_matching = engine.args.run_matching
    run_registration = engine.args.run_registration

    coarse_matching_meter = StatisticsDictMeter()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('recall')
    coarse_matching_meter.register_meter('MR>0')
    coarse_matching_meter.register_meter('MR>=0.1')
    coarse_matching_meter.register_meter('MR>=0.3')
    coarse_matching_meter.register_meter('MR>=0.5')
    coarse_matching_meter.register_meter('scene_precision')
    coarse_matching_meter.register_meter('scene_recall')
    coarse_matching_meter.register_meter('scene_MR>0')
    coarse_matching_meter.register_meter('scene_MR>=0.1')
    coarse_matching_meter.register_meter('scene_MR>=0.3')
    coarse_matching_meter.register_meter('scene_MR>=0.5')

    fine_matching_meter = StatisticsDictMeter()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('num_inlier')
    fine_matching_meter.register_meter('scene_recall')
    fine_matching_meter.register_meter('scene_inlier_ratio')
    fine_matching_meter.register_meter('scene_overlap')

    registration_meter = StatisticsDictMeter()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('mean_rre')
    registration_meter.register_meter('mean_rte')
    registration_meter.register_meter('median_rre')
    registration_meter.register_meter('median_rte')

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_roots = sorted(glob.glob(osp.join(features_root, '*')))
    for scene_root in scene_roots:
        coarse_matching_meter.reset_meter('scene_precision')
        coarse_matching_meter.reset_meter('scene_recall')
        coarse_matching_meter.reset_meter('scene_MR>0')
        coarse_matching_meter.reset_meter('scene_MR>=0.1')
        coarse_matching_meter.reset_meter('scene_MR>=0.3')
        coarse_matching_meter.reset_meter('scene_MR>=0.5')

        fine_matching_meter.reset_meter('scene_recall')
        fine_matching_meter.reset_meter('scene_inlier_ratio')
        fine_matching_meter.reset_meter('scene_overlap')

        estimated_transforms = []

        scene_name = osp.basename(scene_root)
        file_names = sorted(glob.glob(osp.join(scene_root, '*.npz')),
                            key=lambda x: [int(i) for i in osp.basename(x).split('.')[0].split('_')])
        for file_name in file_names:
            id0, id1 = [int(x) for x in osp.basename(file_name).split('.')[0].split('_')]
            data_dict = np.load(file_name)

            ref_points_c = data_dict['ref_points_c']
            src_points_c = data_dict['src_points_c']
            ref_node_corr_indices = data_dict['ref_node_corr_indices']
            src_node_corr_indices = data_dict['src_node_corr_indices']
            gt_node_corr_indices = data_dict['gt_node_corr_indices']

            ref_corr_points = data_dict['ref_corr_points']
            src_corr_points = data_dict['src_corr_points']
            estimated_transform = data_dict['estimated_transform']

            transform = data_dict['transform']
            pcd_overlap = data_dict['overlap']

            # evaluate correspondences
            if run_matching:
                # evaluate coarse correspondences
                coarse_precision, coarse_recall = eval_superpoint_matching(
                    ref_points_c, src_points_c, ref_node_corr_indices, src_node_corr_indices, gt_node_corr_indices
                )

                coarse_matching_meter.update('scene_precision', coarse_precision)
                coarse_matching_meter.update('scene_recall', coarse_recall)
                coarse_matching_meter.update('scene_MR>0', float(coarse_precision > 0))
                coarse_matching_meter.update('scene_MR>=0.1', float(coarse_precision >= 0.1))
                coarse_matching_meter.update('scene_MR>=0.3', float(coarse_precision >= 0.3))
                coarse_matching_meter.update('scene_MR>=0.5', float(coarse_precision >= 0.5))

                # evaluate fine correspondences
                result_dict = evaluate_correspondences(
                    ref_corr_points, src_corr_points, transform, positive_radius=config.test_tau1
                )

                inlier_ratio = result_dict['inlier_ratio']
                overlap = result_dict['overlap']

                fine_matching_meter.update('scene_inlier_ratio', inlier_ratio)
                fine_matching_meter.update('scene_overlap', overlap)
                fine_matching_meter.update('scene_recall', float(inlier_ratio >= config.test_tau2))
                fine_matching_meter.update('num_inlier', inlier_ratio * ref_corr_points.shape[0])

            # evaluate registration
            if run_registration:
                estimated_transforms.append({
                    'test_pair': [id0, id1],
                    'num_fragment': get_num_fragment(scene_name),
                    'transform': estimated_transform
                })

        engine.logger.info('Scene_name: {}'.format(scene_name))

        # print correspondence evaluation results (one scene)
        if run_matching:
            # coarse level statistics
            coarse_precision = coarse_matching_meter.mean('scene_precision')
            coarse_recall = coarse_matching_meter.mean('scene_recall')
            coarse_matching_recall_0 = coarse_matching_meter.mean('scene_MR>0')
            coarse_matching_recall_1 = coarse_matching_meter.mean('scene_MR>=0.1')
            coarse_matching_recall_3 = coarse_matching_meter.mean('scene_MR>=0.3')
            coarse_matching_recall_5 = coarse_matching_meter.mean('scene_MR>=0.5')
            coarse_matching_meter.update('precision', coarse_precision)
            coarse_matching_meter.update('recall', coarse_recall)
            coarse_matching_meter.update('MR>0', coarse_matching_recall_0)
            coarse_matching_meter.update('MR>=0.1', coarse_matching_recall_1)
            coarse_matching_meter.update('MR>=0.3', coarse_matching_recall_3)
            coarse_matching_meter.update('MR>=0.5', coarse_matching_recall_5)
            scene_coarse_matching_result_dict[scene_name] = {
                'precision': coarse_precision,
                'recall': coarse_recall,
                'MR>0': coarse_matching_recall_0,
                'MR>=0.1': coarse_matching_recall_1,
                'MR>=0.3': coarse_matching_recall_3,
                'MR>=0.5': coarse_matching_recall_5
            }

            # fine level statistics
            recall = fine_matching_meter.mean('scene_recall')
            inlier_ratio = fine_matching_meter.mean('scene_inlier_ratio')
            overlap = fine_matching_meter.mean('scene_overlap')
            fine_matching_meter.update('recall', recall)
            fine_matching_meter.update('inlier_ratio', inlier_ratio)
            fine_matching_meter.update('overlap', overlap)
            scene_fine_matching_result_dict[scene_name] = {
                'recall': recall,
                'inlier_ratio': inlier_ratio
            }
            message = '  Correspondence, ' + \
                      'c_prec: {:.3f}, '.format(coarse_precision) + \
                      'c_recall: {:.3f}, '.format(coarse_recall) + \
                      'c_MR>0: {:.3f}, '.format(coarse_matching_recall_0) + \
                      'c_MR>=0.1: {:.3f}, '.format(coarse_matching_recall_1) + \
                      'c_MR>=0.3: {:.3f}, '.format(coarse_matching_recall_3) + \
                      'c_MR>=0.5: {:.3f}, '.format(coarse_matching_recall_5) + \
                      'f_FMR: {:.3f}, '.format(recall) + \
                      'f_IR: {:.3f}, '.format(inlier_ratio) + \
                      'f_OV: {:.3f}'.format(overlap)
            engine.logger.info(message)

        # print registration evaluation results (one scene)
        if run_registration:
            est_log = osp.join(config.registration_dir, benchmark, scene_name, 'est.log')
            gt_log = osp.join(config.dataset_root, 'metadata', 'benchmarks', benchmark, scene_name, 'gt.log')
            gt_info = osp.join(config.dataset_root, 'metadata', 'benchmarks', benchmark, scene_name, 'gt.info')
            write_log_file(est_log, estimated_transforms)

            result_dict = evaluate_registration_3dmatch(gt_log, gt_info, est_log, config.test_registration_threshold)
            registration_meter.update_from_result_dict(result_dict)

            scene_registration_result_dict[scene_name] = {
                'recall': result_dict['recall'],
                'rre': result_dict['median_rre'],
                'rte': result_dict['median_rte']
            }

            message = '  Registration, ' + \
                      'RR: {:.3f}, '.format(result_dict['recall']) + \
                      'mean_RRE: {:.3f}, '.format(result_dict['mean_rre']) + \
                      'mean_RTE: {:.3f}, '.format(result_dict['mean_rte']) + \
                      'median_RRE: {:.3f}, '.format(result_dict['median_rre']) + \
                      'median_RTE: {:.3f}'.format(result_dict['median_rte'])
            engine.logger.info(message)

    engine.logger.critical('Epoch {}'.format(engine.args.test_epoch))

    # print correspondence evaluation results
    if run_matching:
        message = '  Coarse Matching, ' + \
                  'precision: {:.3f}, '.format(coarse_matching_meter.mean('precision')) + \
                  'recall: {:.3f}, '.format(coarse_matching_meter.mean('recall')) + \
                  'MR>0: {:.3f}, '.format(coarse_matching_meter.mean('MR>0')) + \
                  'MR>=0.1: {:.3f}, '.format(coarse_matching_meter.mean('MR>=0.1')) + \
                  'MR>=0.3: {:.3f}, '.format(coarse_matching_meter.mean('MR>=0.3')) + \
                  'MR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('MR>=0.5'))
        engine.logger.critical(message)
        for scene_name, result_dict in scene_coarse_matching_result_dict.items():
            message = '    {}, '.format(get_scene_abbr(scene_name)) + \
                      'precision: {:.3f}, '.format(result_dict['precision']) + \
                      'recall: {:.3f}, '.format(result_dict['recall']) + \
                      'MR>0: {:.3f}, '.format(result_dict['MR>0']) + \
                      'MR>=0.1: {:.3f}, '.format(result_dict['MR>=0.1']) + \
                      'MR>=0.3: {:.3f}, '.format(result_dict['MR>=0.3']) + \
                      'MR>=0.5: {:.3f}'.format(result_dict['MR>=0.5'])
            engine.logger.critical(message)
        message = '  Fine Matching, ' + \
                  'FMR: {:.3f}, '.format(fine_matching_meter.mean('recall')) + \
                  'IR: {:.3f}, '.format(fine_matching_meter.mean('inlier_ratio')) + \
                  'OV: {:.3f}, '.format(fine_matching_meter.mean('overlap')) + \
                  'NI: {:.3f}, '.format(fine_matching_meter.mean('num_inlier')) + \
                  'std: {:.3f}'.format(fine_matching_meter.std('recall'))
        engine.logger.critical(message)
        for scene_name, result_dict in scene_fine_matching_result_dict.items():
            message = '    {}, FMR: {:.3f}'.format(get_scene_abbr(scene_name), result_dict['recall'])
            engine.logger.critical(message)

    # print registration evaluation results
    if run_registration:
        message = '  Registration, ' + \
                  'RR: {:.3f}, '.format(registration_meter.mean('recall')) + \
                  'mean_RRE: {:.3f}, '.format(registration_meter.mean('mean_rre')) + \
                  'mean_RTE: {:.3f}, '.format(registration_meter.mean('mean_rte')) + \
                  'median_RRE: {:.3f}, '.format(registration_meter.mean('median_rre')) + \
                  'median_RTE: {:.3f}'.format(registration_meter.mean('median_rte'))
        engine.logger.critical(message)
        for scene_name, result_dict in scene_registration_result_dict.items():
            message = '    {}, RR: {:.3f}, RRE: {:.3f}, RTE: {:.3f}'.format(
                get_scene_abbr(scene_name), result_dict['recall'], result_dict['rre'], result_dict['rte']
            )
            engine.logger.critical(message)


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        message = 'Epoch: {}, benchmark: {}'.format(engine.args.test_epoch, engine.args.benchmark)
        engine.logger.critical(message)
        message = 'Superpoint matching config, ' + \
                  'num_proposal: {}, '.format(config.superpoint_matching_num_proposal) + \
                  'dual_normalization: {}, '.format(config.superpoint_matching_dual_normalization)
        engine.logger.critical(message)
        message = 'Point matching config, ' + \
                  'k: {}, '.format(config.point_matching_topk) + \
                  'threshold: {:.2f}, '.format(config.point_matching_confidence_threshold) + \
                  'positive_radius: {:.2f}, '.format(config.point_matching_positive_radius) + \
                  'min_num_corr: {}, '.format(config.point_matching_min_num_corr) + \
                  'max_num_corr: {}, '.format(config.point_matching_max_num_corr) + \
                  'num_registration_iter: {}'.format(config.point_matching_num_registration_iter)
        engine.logger.critical(message)

        if not engine.args.run_matching and not engine.args.run_registration:
            engine.logger.info('Flags "--run_matching" and "--run_registration" are both not set, skipped.')
        else:
            eval_one_epoch(engine)


if __name__ == '__main__':
    main()
