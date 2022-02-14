import os
import os.path as osp


def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def safe_divide(a, b):
    if b == 0:
        return 0
    else:
        return a / b
