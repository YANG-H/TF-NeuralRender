import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL = tf.load_op_library(os.path.join(
    BASE_DIR, 'qt_build_release', 'libTFNeuralRenderOps.so'))


def rasterize(pts, faces, uvs, H, W):
    """
    Input:
    ---
        - pts: BxNpx3 float32
        - faces:  BxNfx3 int32
        - uvs: BxNfx3x2 float32
        - screen_shape: [H W]

    Output:
    ---
        - uvgrid: BxHxWx2
        - fids: BxHxW
        - z: BxHxW
    """
    assert pts.shape[0] == faces.shape[0] and pts.shape[0] == uvs.shape[0]
    assert pts.shape[2] == 3
    assert faces.shape[1] == uvs.shape[1]
    assert faces.shape[2] == 3
    assert uvs.shape[2] == 3 and uvs.shape[3] == 2
    uvgrid, fids, z = MODEL.rasterize(pts, faces, uvs, H=H, W=W)
    return uvgrid, fids, z


ops.NoGradient('Rasterize')
