import numpy as np


src = np.random.rand(3, 68, 2)
dst = np.random.rand(3, 68, 2)


def calc_nme(src, dst):

    bbox = np.max(dst, axis=-2) - np.min(dst, axis=-2)
    euler = np.sqrt(np.sum((src - dst)**2, axis=-1))
    w = 1 / np.sqrt(bbox[:, 0] * bbox[:, 1])
    nme = w * np.mean(euler, axis=-1)
    return nme

def calc_auc():
    pass

calc_nme(src, dst)