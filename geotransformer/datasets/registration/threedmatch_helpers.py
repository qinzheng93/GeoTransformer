import os.path as osp

import numpy as np
from nibabel import quaternions as nq

from ...utils.point_cloud_utils import get_rotation_translation_from_transform
from ...utils.registration_utils import compute_registration_error
from ...utils.metrics import StatisticsMeter
from ...utils.python_utils import ensure_dir


_scene_name_to_num_fragment = {
    '7-scenes-redkitchen': 60,
    'sun3d-home_at-home_at_scan1_2013_jan_1': 60,
    'sun3d-home_md-home_md_scan9_2012_sep_30': 60,
    'sun3d-hotel_uc-scan3': 55,
    'sun3d-hotel_umd-maryland_hotel1': 57,
    'sun3d-hotel_umd-maryland_hotel3': 37,
    'sun3d-mit_76_studyroom-76-1studyroom2': 66,
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 38
}


_scene_name_to_abbr = {
    '7-scenes-redkitchen': 'Kitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1': 'Home_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30': 'Home_2',
    'sun3d-hotel_uc-scan3': 'Hotel_1',
    'sun3d-hotel_umd-maryland_hotel1': 'Hotel_2',
    'sun3d-hotel_umd-maryland_hotel3': 'Hotel_3',
    'sun3d-mit_76_studyroom-76-1studyroom2': 'Study',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 'MIT_Lab'
}


def get_num_fragment(scene_name):
    if scene_name not in _scene_name_to_num_fragment:
        raise ValueError('Unsupported test scene name "{}".'.format(scene_name))
    return _scene_name_to_num_fragment[scene_name]


def get_scene_abbr(scene_name):
    if scene_name not in _scene_name_to_abbr:
        return scene_name
    else:
        return _scene_name_to_abbr[scene_name]


def read_log_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pair = len(lines) // 5
    for i in range(num_pair):
        line_id = i * 5
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragment = int(split_line[2])
        transform = []
        for j in range(1, 5):
            transform.append(lines[line_id + j].split())
        transform = np.array(transform, dtype=np.float32)
        test_pairs.append({
            'test_pair': test_pair,
            'num_fragment': num_fragment,
            'transform': transform
        })
        '''
        transform is the pose from test_pair[1] to test_pair[0]
        '''
    return test_pairs


def read_info_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pair = len(lines) // 7
    for i in range(num_pair):
        line_id = i * 7
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragment = int(split_line[2])
        info = []
        for j in range(1, 7):
            info.append(lines[line_id + j].split())
        info = np.array(info, dtype=np.float32)
        test_pairs.append({
            'test_pair': test_pair,
            'num_fragment': num_fragment,
            'covariance': info
        })
    return test_pairs


def write_log_file(file_name, test_pairs):
    ensure_dir(osp.dirname(file_name))
    lines = []
    for test_pair in test_pairs:
        frag_id0, frag_id1 = test_pair['test_pair']
        lines.append('{}\t{}\t{}\n'.format(frag_id0, frag_id1, test_pair['num_fragment']))
        rows = test_pair['transform'].tolist()
        for row in rows:
            lines.append('{}\t{}\t{}\t{}\n'.format(row[0], row[1], row[2], row[3]))

    with open(file_name, 'w') as f:
        f.writelines(lines)


def get_gt_logs_and_infos(gt_root, num_fragment):
    gt_logs = read_log_file(osp.join(gt_root, 'gt.log'))
    gt_infos = read_info_file(osp.join(gt_root, 'gt.info'))

    gt_indices = -np.ones((num_fragment, num_fragment), dtype=np.int32)
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


def evaluate_registration_3dmatch(gt_log_file, gt_info_file, result_file, positive_threshold=0.2):
    rre_meter = StatisticsMeter()
    rte_meter = StatisticsMeter()

    gt_logs = read_log_file(gt_log_file)
    gt_infos = read_info_file(gt_info_file)
    result_logs = read_log_file(result_file)

    num_fragment = gt_logs[0]['num_fragment']
    num_correct = 0
    num_ground_truth = 0
    num_prediction = 0

    gt_indices = -np.ones((num_fragment, num_fragment), dtype=np.int32)
    for i, gt_log in enumerate(gt_logs):
        frag_id0, frag_id1 = gt_log['test_pair']
        if frag_id1 > frag_id0 + 1:
            gt_indices[frag_id0, frag_id1] = i
            num_ground_truth += 1

    errors = []

    for result_log in result_logs:
        frag_id0, frag_id1 = result_log['test_pair']
        estimated_transform = result_log['transform']
        if gt_indices[frag_id0, frag_id1] != -1:
            num_prediction += 1

            gt_index = gt_indices[frag_id0, frag_id1]
            transform = gt_logs[gt_index]['transform']
            covariance = gt_infos[gt_index]['covariance']
            assert gt_infos[gt_index]['test_pair'][0] == frag_id0 and gt_infos[gt_index]['test_pair'][1] == frag_id1
            error = compute_transform_error(transform, covariance, estimated_transform)

            errors.append({
                'id0': frag_id0,
                'id1': frag_id1,
                'error': error
            })

            if error <= positive_threshold ** 2:
                num_correct += 1
                relative_rotation_error, relative_translation_error = \
                    compute_registration_error(transform, estimated_transform)
                rre_meter.update(relative_rotation_error)
                rte_meter.update(relative_translation_error)

    precision = num_correct / num_prediction if num_prediction > 0 else 0
    recall = num_correct / num_ground_truth

    result_dict = {
        'precision': precision,
        'recall': recall,
        'mean_rre': rre_meter.mean(),
        'mean_rte': rte_meter.mean(),
        'median_rre': rre_meter.median(),
        'median_rte': rte_meter.median(),
        'num_correct': num_correct,
        'num_prediction': num_prediction,
        'num_ground_truth': num_ground_truth,
        'errors': errors
    }

    return result_dict
