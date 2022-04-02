import h5py
import numpy as np
import pickle


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def process(subset):
    with open(f'modelnet40_ply_hdf5_2048/{subset}_files.txt') as f:
        lines = f.readlines()
    all_points = []
    all_normals = []
    all_labels = []
    for line in lines:
        filename = line.strip()
        h5file = h5py.File(f'modelnet40_ply_hdf5_2048/{filename}', 'r')
        all_points.append(h5file['data'][:])
        all_normals.append(h5file['normal'][:])
        all_labels.append(h5file['label'][:].flatten().astype(np.int))
    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f'{subset} data loaded.')
    all_data = []
    num_data = points.shape[0]
    for i in range(num_data):
        all_data.append(dict(points=points[i], normals=normals[i], label=labels[i]))
    if subset == 'train':
        indices = np.random.permutation(num_data)
        num_train = int(num_data * 0.8)
        num_val = num_data - num_train
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        train_data = [all_data[i] for i in train_indices.tolist()]
        dump_pickle(train_data, 'train.pkl')
        val_data = [all_data[i] for i in val_indices.tolist()]
        dump_pickle(val_data, 'val.pkl')
    else:
        dump_pickle(all_data, 'test.pkl')



for subset in ['train', 'test']:
    process(subset)
