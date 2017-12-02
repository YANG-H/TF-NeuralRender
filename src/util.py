import numpy as np

import time
import logging


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
        print(elapsed)
        return result

    return wrap
