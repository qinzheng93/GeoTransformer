import os.path as osp

import numpy as np
from nibabel import quaternions as nq

from geotransformer.utils.common import ensure_dir
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.pointcloud import (
    apply_transform,
    get_rotation_translation_from_transform,
    get_nearest_neighbor,
)
from geotransformer.utils.registration import compute_overlap, compute_registration_error

_scene_name_to_num_fragments = {
    '7-scenes-redkitchen': 60,
    'sun3d-home_at-home_at_scan1_2013_jan_1': 60,
    'sun3d-home_md-home_md_scan9_2012_sep_30': 60,
    'sun3d-hotel_uc-scan3': 55,
    'sun3d-hotel_umd-maryland_hotel1': 57,
    'sun3d-hotel_umd-maryland_hotel3': 37,
    'sun3d-mit_76_studyroom-76-1studyroom2': 66,
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 38,
}


_scene_name_to_abbr = {
    '7-scenes-redkitchen': 'Kitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1': 'Home_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30': 'Home_2',
    'sun3d-hotel_uc-scan3': 'Hotel_1',
    'sun3d-hotel_umd-maryland_hotel1': 'Hotel_2',
    'sun3d-hotel_umd-maryland_hotel3': 'Hotel_3',
    'sun3d-mit_76_studyroom-76-1studyroom2': 'Study',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 'MIT_Lab',
}


def get_num_fragments(scene_name):
    if scene_name not in _scene_name_to_num_fragments:
        raise ValueError('Unsupported test scene name "{}".'.format(scene_name))
    return _scene_name_to_num_fragments[scene_name]


def get_scene_abbr(scene_name):
    if scene_name not in _scene_name_to_abbr:
        return scene_name
    else:
        return _scene_name_to_abbr[scene_name]


def read_pose_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    pose = []
    for line in lines:
        pose_row = [float(x) for x in line.strip().split()]
        pose.append(pose_row)
    pose = np.stack(pose, axis=0)
    return pose


def read_log_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 5
    for i in range(num_pairs):
        line_id = i * 5
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        transform = []
        for j in range(1, 5):
            transform.append(lines[line_id + j].split())
        # transform is the pose from test_pair[1] to test_pair[0]
        transform = np.array(transform, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, transform=transform))
    return test_pairs


def read_info_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 7
    for i in range(num_pairs):
        line_id = i * 7
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        info = []
        for j in range(1, 7):
            info.append(lines[line_id + j].split())
        info = np.array(info, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, covariance=info))
    return test_pairs


def write_log_file(file_name, test_pairs):
    ensure_dir(osp.dirname(file_name))
    lines = []
    for test_pair in test_pairs:
        frag_id0, frag_id1 = test_pair['test_pair']
        lines.append('{}\t{}\t{}\n'.format(frag_id0, frag_id1, test_pair['num_fragments']))
        rows = test_pair['transform'].tolist()
        for row in rows:
            lines.append('{}\t{}\t{}\t{}\n'.format(row[0], row[1], row[2], row[3]))

    with open(file_name, 'w') as f:
        f.writelines(lines)


def get_gt_logs_and_infos(gt_root, num_fragments):
    gt_logs = read_log_file(osp.join(gt_root, 'gt.log'))
    gt_infos = read_info_file(osp.join(gt_root, 'gt.info'))

    gt_indices = -np.ones((num_fragments, num_fragments), dtype=np.int32)
    for i, gt_log in enumerate(gt_logs):
        frag_id0, frag_id1 = gt_log['test_pair']
        if frag_id1 > frag_id0 + 1:
            gt_indices[frag_id0, frag_id1] = i

    return gt_indices, gt_logs, gt_infos


def compute_transform_error(transform, covariance, estimated_transform):
    relative_transform = np.matmul(np.linalg.inv(transform), estimated_transform)
    R, t = get_rotation_translation_from_transform(relative_transform)
    q = nq.mat2quat(R)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]
    return p.item()


def evaluate_registration_one_scene(gt_log_file, gt_info_file, result_file, positive_threshold=0.2):
    registration_meter = SummaryBoard(['rre', 'rte'])

    gt_logs = read_log_file(gt_log_file)
    gt_infos = read_info_file(gt_info_file)
    result_logs = read_log_file(result_file)

    num_fragments = gt_logs[0]['num_fragments']
    num_pos_pairs = 0
    num_gt_pairs = 0
    num_pred_pairs = 0

    gt_indices = -np.ones((num_fragments, num_fragments), dtype=np.int32)
    for i, gt_log in enumerate(gt_logs):
        frag_id0, frag_id1 = gt_log['test_pair']
        if frag_id1 > frag_id0 + 1:
            gt_indices[frag_id0, frag_id1] = i
            num_gt_pairs += 1

    errors = []

    for result_log in result_logs:
        frag_id0, frag_id1 = result_log['test_pair']
        estimated_transform = result_log['transform']
        if gt_indices[frag_id0, frag_id1] != -1:
            num_pred_pairs += 1

            gt_index = gt_indices[frag_id0, frag_id1]
            transform = gt_logs[gt_index]['transform']
            covariance = gt_infos[gt_index]['covariance']
            assert gt_infos[gt_index]['test_pair'][0] == frag_id0 and gt_infos[gt_index]['test_pair'][1] == frag_id1
            error = compute_transform_error(transform, covariance, estimated_transform)

            errors.append({'id0': frag_id0, 'id1': frag_id1, 'error': error})

            if error <= positive_threshold ** 2:
                num_pos_pairs += 1
                rre, rte = compute_registration_error(transform, estimated_transform)
                registration_meter.update('rre', rre)
                registration_meter.update('rte', rte)

    precision = num_pos_pairs / num_pred_pairs if num_pred_pairs > 0 else 0
    recall = num_pos_pairs / num_gt_pairs

    return {
        'precision': precision,
        'recall': recall,
        'mean_rre': registration_meter.mean('rre'),
        'mean_rte': registration_meter.mean('rte'),
        'median_rre': registration_meter.median('rre'),
        'median_rte': registration_meter.median('rte'),
        'num_pos_pairs': num_pos_pairs,
        'num_pred_pairs': num_pred_pairs,
        'num_gt_pairs': num_gt_pairs,
        'errors': errors,
    }


def calibrate_ground_truth(ref_pcd, src_pcd, transform, voxel_size=0.006):
    ref_pcd = ref_pcd.voxel_down_sample(0.01)
    src_pcd = src_pcd.voxel_down_sample(0.01)
    ref_points = np.asarray(ref_pcd.points)
    src_points = np.asarray(src_pcd.points)

    # compute overlap
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=voxel_size * 5)

    # compute info
    src_points = apply_transform(src_points, transform)
    nn_distances, nn_indices = get_nearest_neighbor(ref_points, src_points, return_index=True)
    nn_indices = nn_indices[nn_distances < voxel_size]
    if nn_indices.shape[0] > 5000:
        nn_indices = np.random.choice(nn_indices, 5000, replace=False)
    src_corr_points = src_points[nn_indices]
    if src_corr_points.shape[0] > 0:
        g = np.zeros([src_corr_points.shape[0], 3, 6])
        g[:, :3, :3] = np.eye(3)
        g[:, 0, 4] = src_corr_points[:, 2]
        g[:, 0, 5] = -src_corr_points[:, 1]
        g[:, 1, 3] = -src_corr_points[:, 2]
        g[:, 1, 5] = src_corr_points[:, 0]
        g[:, 2, 3] = src_corr_points[:, 1]
        g[:, 2, 4] = -src_corr_points[:, 0]
        gt = g.transpose([0, 2, 1])
        gtg = np.matmul(gt, g)
        cov_matrix = gtg.sum(0)
    else:
        cov_matrix = np.zeros((6, 6))

    return overlap, cov_matrix
