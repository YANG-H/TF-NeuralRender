import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def rasterize(pts, faces, uvs, tex, H, W):
    """
    Input:
    ---
        - pts: BxNpx4 float32
        - faces:  BxNfx3 int32
        - uvs: BxNfx3x2 float32
        - tex: BxHxWx4 float32
        - H: height
        - W: width

    Output:
    ---
        - renderings: BxHxWx4
    """

    pass
