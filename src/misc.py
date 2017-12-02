import os
import sys
import time
import logging

import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def normalize(v):
    return v / np.expand_dims(np.linalg.norm(v, axis=-1), axis=-1)


def transform(mat, pts, out4d=False):
    pts = np.concatenate((pts, np.ones([pts.shape[0], 1])), axis=1)
    pts = np.matmul(pts, np.transpose(mat))
    return pts if out4d else pts[:, 0:3] / np.expand_dims(pts[:, 3], axis=-1)


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - started_at
        logging.info(elapsed)
        print('#time cost: %f seconds' % elapsed)
        return result

    return wrap
