import os
import sys
import numpy as np
import tensorflow as tf
import neural_render as nr
import spatial_transform as st
import pymesh as pm
import matplotlib.pyplot as plt
from scipy import ndimage


import spatial_transform as st
import neural_render as nr
import camera
import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


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
    '''
    pts = camera.apply_transform(pts, modelview, proj)
    uvgrid, z, fids, bc = nr.rasterize(pts, faces, uvs, H, W)
    rendered = st.bilinear_sampler(tex, uvgrid[:, :, :, 0], uvgrid[:, :, :, 1])
    return rendered


def main():
    mesh = pm.meshutils.generate_icosphere(
        radius=1, center=np.array([0, 0, 0]))
    pts = np.expand_dims(mesh.vertices, 0).astype('float32')
    faces = np.expand_dims(mesh.faces, 0).astype('int32')
    uvs = np.array([[[[0, 0], [0, 1], [1, 0]]]], dtype='float32')
    uvs = np.tile(uvs, reps=faces.shape[:2] + (1, 1))
    tex = ndimage.imread(os.path.join(BASE_DIR, 'data', 'chessboard.jpg'))
    tex = np.expand_dims(tex, axis=0).astype('float32') / 255.0

    with tf.Session().as_default():
        mv = camera.look_at(eye=tf.constant([2, 3, 3.2], dtype='float32'),
                            center=tf.constant([0, 0, 0], dtype='float32'),
                            up=tf.constant([0, 0, -1], dtype='float32'))
        mv = tf.expand_dims(mv, axis=0)
        H = 600
        W = 600
        proj = camera.perspective(focal=600, H=H, W=W)
        proj = tf.expand_dims(proj, axis=0)

        n = 4
        rendered = render(pts=tf.tile(pts, [n, 1, 1]),
                          faces=tf.tile(faces, [n, 1, 1]),
                          uvs=tf.tile(uvs, [n, 1, 1, 1]),
                          tex=tf.tile(tex, [n, 1, 1, 1]),
                          modelview=tf.tile(mv, [n, 1, 1]),
                          proj=tf.tile(proj, [n, 1, 1]), H=H, W=W).eval()

    plt.figure()
    plt.imshow(rendered[3, :, :, :])
    plt.show()


if __name__ == '__main__':
    main()
