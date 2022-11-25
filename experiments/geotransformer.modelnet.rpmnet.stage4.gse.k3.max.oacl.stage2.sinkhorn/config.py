import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, "data", "ModelNet")
_C.data.num_points = 717
_C.data.voxel_size = None
_C.data.rotation_magnitude = 45.0
_C.data.translation_magnitude = 0.5
_C.data.keep_ratio = 0.7
_C.data.crop_method = "plane"
_C.data.asymmetric = True
_C.data.twice_sample = True
_C.data.twice_transform = False

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.noise_magnitude = 0.05
_C.train.class_indices = "all"

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.noise_magnitude = 0.05
_C.test.class_indices = "all"

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 1.0
_C.eval.rte_threshold = 0.1

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 10000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1
_C.optim.max_iteration = 400000
_C.optim.snapshot_steps = 10000
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 3
_C.backbone.init_voxel_size = 0.05
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 128
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
