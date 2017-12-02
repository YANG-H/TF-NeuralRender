import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

import misc
import spatial_transform as st
import camera

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL = tf.load_op_library(os.path.join(
    BASE_DIR, 'qt_build_release', 'libTFNeuralRenderOps.so'))


@misc.profile
def rasterize(pts, faces, uvs, H=400, W=400):
    ''' rasterize
    Input
    ---
    - `pts`: BxNpx3 float32
    - `faces`:  BxNfx3 int32
    - `uvs`: BxNfx3x2 float32
    
    Output
    ---
    - `uvgrid`: BxHxWx2
    - `z`: BxHxW
    - `fids`: BxHxW
    - `bc`: BxHxWx3, barycentric coordinates
    '''
    assert pts.shape[0] == faces.shape[0] and pts.shape[0] == uvs.shape[0]
    assert pts.shape[2] == 3
    assert faces.shape[1] == uvs.shape[1]
    assert faces.shape[2] == 3
    assert uvs.shape[2] == 3 and uvs.shape[3] == 2
    uvgrid, z, fids, bc = MODEL.rasterize(pts, faces, uvs, H=H, W=W)
    return uvgrid, z, fids, bc


@tf.RegisterGradient('Rasterize')
def _rasterize_grad(op, grad_uvgrid, grad_z, grad_fids, grad_bc):
    pts, faces, uvs = op.inputs
    uvgrid, z, fids, bc = op.outputs
    grad_pts = MODEL.rasterize_grad(
        pts, faces, uvs, uvgrid, z, fids, bc, grad_uvgrid, grad_z, grad_bc)
    return grad_pts, None, None


def render(pts, faces, uvs, tex, modelview, proj, H, W):
    ''' render textured mesh to image
    Input
    ---
    - `pts`: BxNpx3, float32
    - `faces`: BxNfx3, int32
    - `uvs`: BxNfx3x2, float32
    - `tex`: BxHtxWtxD, float32
    - `modelview`: Bx4x4, float32
    - `proj`: Bx4x4, float32
    
    Output
    ---
    - `rendered`: BxHxWxD, float32
    - `uvgrid`: BxHxWx2
    - `z`: BxHxW
    - `fids`: BxHxW
    - `bc`: BxHxWx3, barycentric coordinates
    '''
    pts = camera.apply_transform(pts, modelview, proj)
    uvgrid, z, fids, bc = rasterize(pts, faces, uvs, H, W)
    rendered = st.bilinear_sampler(tex, uvgrid[:, :, :, 0], uvgrid[:, :, :, 1])
    return rendered, uvgrid, z, fids, bc
