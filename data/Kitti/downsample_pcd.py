import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm


def main():
    for i in range(11):
        seq_id = '{:02d}'.format(i)
        file_names = glob.glob(osp.join('sequences', seq_id, 'velodyne', '*.bin'))
        for file_name in tqdm(file_names):
            frame = file_name.split('/')[-1][:-4]
            new_file_name = osp.join('downsampled', seq_id, frame + '.npy')
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(0.3)
            points = np.array(pcd.points).astype(np.float32)
            np.save(new_file_name, points)


if __name__ == '__main__':
    main()
