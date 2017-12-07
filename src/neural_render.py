import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

import misc
import spatial_transform as st
import camera

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = tf.load_op_library(os.path.join(
    BASE_DIR, 'build_release', 'libTFNeuralRenderOps.so'))


def make_uvs(nfaces):
    uvs = tf.stack([tf.range(1, nfaces + 1, dtype=tf.float32) / (nfaces),
                    tf.zeros([nfaces], dtype=tf.float32) + 0.1,
                    tf.range(1, nfaces + 1, dtype=tf.float32) / (nfaces),
                    tf.ones([nfaces], dtype=tf.float32) - 0.1,
                    tf.range(0, nfaces, dtype=tf.float32) / (nfaces),
                    tf.ones([nfaces], dtype=tf.float32) - 0.1])
    uvs = tf.transpose(uvs, [1, 0])  # nfaces x 6
    uvs = tf.reshape(uvs, [nfaces, 3, 2])  # nfaces x 3 x 2
    return uvs


def rasterize(pts, faces, uvs, H=400, W=400, **kwargs):
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
    uvgrid, z, fids, bc = MODEL.rasterize(pts, faces, uvs, H=H, W=W, **kwargs)
    return uvgrid, z, fids, bc


@tf.RegisterGradient('Rasterize')
def _rasterize_grad(op, grad_uvgrid, grad_z, grad_fids, grad_bc, **kwargs):
    pts, faces, uvs = op.inputs
    _, _, fids, bc = op.outputs
    grad_pts = MODEL.rasterize_grad(
        pts, faces, uvs, fids, bc, grad_uvgrid, grad_z, **kwargs)
    return grad_pts, None, None


def bilinear_sample(tex, uvgrid):
    ''' bilinear_sample
    Input
    ---
    - `tex`: BxHtxWtxDt float32
    - `uvgrid`:  BxHxWx2 float32

    Output
    ---
    - `sampled`: BxHxWxDt float32
    '''
    assert tex.shape[0] == uvgrid.shape[0]
    return MODEL.rasterize(tex, uvgrid)


@tf.RegisterGradient('BilinearSample')
def _bilinear_sample_grad(op, grad_sampled, **kwargs):
    tex, uvgrid = op.inputs
    grad_tex, grad_uvgrid = MODEL.bilinear_sample_grad(tex, uvgrid, grad_sampled)
    return grad_tex, grad_uvgrid


def render(pts, faces, uvs, tex, modelview, proj, H=400, W=400):
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
    uvgrid, z, fids, bc = MODEL.rasterize(pts, faces, uvs, H, W)
    rendered = MODEL.bilinear_sample(tex, uvgrid)
    return rendered, uvgrid, z, fids, bc
