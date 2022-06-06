import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.python_utils import ensure_dir

config = edict()

# random seed
config.seed = 7351

# dir

config.working_dir = osp.dirname(osp.realpath(__file__))
config.program_name = osp.basename(config.working_dir)
config.root_dir = osp.dirname(osp.dirname(config.working_dir))
config.output_dir = osp.join(config.root_dir, 'output', config.program_name)
config.snapshot_dir = osp.join(config.output_dir, 'snapshots')
config.logs_dir = osp.join(config.output_dir, 'logs')
config.features_dir = osp.join(config.output_dir, 'features')
config.registration_dir = osp.join(config.output_dir, 'registration')
# config.dataset_root = osp.join(config.root_dir, 'data', '3DMatch')
config.dataset_root = f'/root/aiyang/Coarse-to-fine-correspondences/scripts'

ensure_dir(config.output_dir)
ensure_dir(config.snapshot_dir)
ensure_dir(config.logs_dir)
ensure_dir(config.features_dir)
ensure_dir(config.registration_dir)

# data
config.voxel_size = 0.025

# train config
config.train_batch_size = 1
config.train_num_worker = 8
config.train_max_num_point = 30000
config.train_use_augmentation = True
config.train_augmentation_noise = 0.005
config.train_rotation_factor = 1.0

# test config
config.test_batch_size = 1
config.test_num_worker = 1
config.test_max_num_point = 30000
config.test_tau1 = 0.1
config.test_tau2 = 0.05
config.test_registration_threshold = 0.2

# optim config
config.learning_rate = 1e-4
config.gamma = 0.95
config.momentum = 0.98
config.weight_decay = 1e-6
config.max_epoch = 40

# model - KPFCNN
config.num_layers = 4
config.in_points_dim = 3
config.first_feats_dim = 128
config.final_feats_dim = 256
config.first_subsampling_dl = 0.025
config.in_features_dim = 1
config.conv_radius = 2.5
config.deform_radius = 5.0
config.num_kernel_points = 15
config.KP_extent = 2.0
config.KP_influence = 'linear'
config.aggregation_mode = 'sum'
config.fixed_kernel_points = 'center'
config.normalization = 'group_norm'
config.normalization_momentum = 0.02
config.deformable = False
config.modulated = False

# model - Architecture
config.architecture = ['simple', 'resnetb']
for i in range(config.num_layers - 1):
    config.architecture.append('resnetb_strided')
    config.architecture.append('resnetb')
    config.architecture.append('resnetb')
for i in range(config.num_layers - 3):
    config.architecture.append('nearest_upsample')
    config.architecture.append('unary')
config.architecture.append('nearest_upsample')
config.architecture.append('last_unary')

# model - Global
config.ground_truth_positive_radius = 0.05
config.point_to_node_max_point = 64
config.sinkhorn_num_iter = 100

config.superpoint_matching_num_target = 128
config.superpoint_matching_overlap_thresh = 0.1
config.superpoint_matching_num_proposal = 256
config.superpoint_matching_dual_normalization = True
config.superpoint_matching_positive_overlap = 0.

config.point_matching_topk = 3
config.point_matching_confidence_threshold = 0.05
config.point_matching_positive_radius = 0.1
config.point_matching_min_num_corr = 3
config.point_matching_max_num_corr = 1500
config.point_matching_num_registration_iter = 5

# # model - Coarse level
# config.geometric_transformer_feats_dim = 256
# config.geometric_transformer_num_head = 4
# config.geometric_transformer_architecture = ['self', 'cross', 'self', 'cross', 'self', 'cross']
# config.geometric_transformer_sigma_d = 0.2
# config.geometric_transformer_sigma_a = 15
# config.geometric_transformer_angle_k = 3

# model - information_interactive
config.information_interactive_nets = ['gge','cross_attn','gge']
config.information_interactive_gnn_feats_dim = 256
config.information_interactive_dgcnn_k = 10
config.information_interactive_ppf_k = 64
config.information_interactive_radius_mul = 32
config.information_interactive_bottleneck = False
config.information_interactive_num_head = 4

# loss - Coarse level
config.overlap_aware_circle_loss_positive_margin = 0.1
config.overlap_aware_circle_loss_negative_margin = 1.4
config.overlap_aware_circle_loss_positive_optimal = 0.1
config.overlap_aware_circle_loss_negative_optimal = 1.4
config.overlap_aware_circle_loss_log_scale = 24
config.overlap_aware_circle_loss_positive_threshold = 0.1

# loss - Fine level
config.point_matching_loss_positive_radius = 0.05

# loss - Overall
config.weight_superpoint_loss = 1.
config.weight_point_loss = 1.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.link_output:
        os.symlink(config.output_dir, 'output')
