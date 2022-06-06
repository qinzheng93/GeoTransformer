import argparse
import os
import os.path as osp
import time
import glob

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from IPython import embed

from geotransformer.utils.metrics import StatisticsDictMeter, Timer
from geotransformer.engine import Engine
from geotransformer.utils.torch_utils import to_cuda
from geotransformer.utils.python_utils import ensure_dir

from dataset import test_data_loader
from config import config
from model import create_model
from loss import Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--test_epoch', type=int, default=5, help='test epoch')
    # parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch'], help='test benchmark')
    parser.add_argument('--benchmark', type=str, default='3DLoMatch', help='test benchmark')
    return parser


def test_one_epoch(engine, epoch, data_loader, model, evaluator):
    features_root = osp.join(config.features_dir, engine.args.benchmark)
    ensure_dir(features_root)

    model.eval()

    result_meter = StatisticsDictMeter()
    result_meter.register_meter('PIR')
    result_meter.register_meter('IR')
    result_meter.register_meter('RRE')
    result_meter.register_meter('RTE')
    result_meter.register_meter('RR')
    timer = Timer()

    num_iter_per_epoch = len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=num_iter_per_epoch)
    for i, data_dict in pbar:
        data_dict = to_cuda(data_dict)

        transform = data_dict['transform']
        scene_name = data_dict['scene_name']
        id0 = data_dict['frag_id0']
        id1 = data_dict['frag_id1']
        overlap = data_dict['overlap']

        with torch.no_grad():
            torch.cuda.synchronize()
            timer.reset_time()
            output_dict = model(data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            result_dict = evaluator(output_dict, data_dict)
            result_dict = {key: value.item() for key, value in result_dict.items()}
            result_meter.update_from_result_dict(result_dict)
            accepted = result_dict['RRE'] < 15. and result_dict['RTE'] < 0.3
            result_meter.update('RR', float(accepted))

        ref_points_c = output_dict['ref_points_c']
        src_points_c = output_dict['src_points_c']
        ref_points_m = output_dict['ref_points_m']
        src_points_m = output_dict['src_points_m']
        ref_points_f = output_dict['ref_points_f']
        src_points_f = output_dict['src_points_f']
        ref_feats_c = output_dict['ref_feats_c']
        src_feats_c = output_dict['src_feats_c']
        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        corr_scores = output_dict['corr_scores']
        estimated_transform = output_dict['estimated_transform']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']

        message = 'Epoch {}, '.format(epoch) + \
                  'iter {}/{}, '.format(i + 1, num_iter_per_epoch) + \
                  '{}, id0: {}, id1: {}, '.format(scene_name, id0, id1) + \
                  'PIR: {:.3f}, '.format(result_dict['PIR']) + \
                  'IR: {:.3f}, '.format(result_dict['IR']) + \
                  'RRE: {:.3f}, '.format(result_dict['RRE']) + \
                  'RTE: {:.3f}, '.format(result_dict['RTE']) + \
                  'n_corr: {}, '.format(corr_scores.shape[0]) + \
                  'time: {:.3f}'.format(timer.get_process_time())
        pbar.set_description(message)

        ensure_dir(osp.join(features_root, scene_name))
        file_name = osp.join(features_root, scene_name, '{}_{}.npz'.format(id0, id1))
        np.savez_compressed(
            file_name,
            ref_points_f=ref_points_f.detach().cpu().numpy(),
            src_points_f=src_points_f.detach().cpu().numpy(),
            ref_points_m=ref_points_m.detach().cpu().numpy(),
            src_points_m=src_points_m.detach().cpu().numpy(),
            ref_points_c=ref_points_c.detach().cpu().numpy(),
            src_points_c=src_points_c.detach().cpu().numpy(),
            ref_feats_c=ref_feats_c.detach().cpu().numpy(),
            src_feats_c=src_feats_c.detach().cpu().numpy(),
            ref_node_corr_indices=ref_node_corr_indices.detach().cpu().numpy(),
            src_node_corr_indices=src_node_corr_indices.detach().cpu().numpy(),
            ref_corr_points=ref_corr_points.detach().cpu().numpy(),
            src_corr_points=src_corr_points.detach().cpu().numpy(),
            corr_scores=corr_scores.detach().cpu().numpy(),
            gt_node_corr_indices=gt_node_corr_indices.detach().cpu().numpy(),
            gt_node_corr_overlaps=gt_node_corr_overlaps.detach().cpu().numpy(),
            estimated_transform=estimated_transform.detach().cpu().numpy(),
            transform=transform.detach().cpu().numpy(),
            overlap=overlap
        )

    message = 'Epoch {}, {}'.format(epoch, result_meter.summary())
    engine.logger.info(message)


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
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

        start_time = time.time()
        data_loader = test_data_loader(config, engine.args.benchmark)
        loading_time = time.time() - start_time

        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model)

        if engine.args.snapshot is not None:
            snapshot = engine.args.snapshot
        elif engine.args.test_epoch is not None:
            snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(engine.args.test_epoch))
        else:
            raise ValueError('Please specify the snapshot to test with "--snapshot" or "--test_epoch".')
        engine.load_snapshot(snapshot)

        test_one_epoch(engine, engine.state.epoch, data_loader, model, evaluator)


if __name__ == '__main__':
    main()
