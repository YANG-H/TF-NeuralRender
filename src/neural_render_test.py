import os
import sys
import numpy as np
import tensorflow as tf
import neural_render as nr
import spatial_transform as st
import pymesh as pm
import matplotlib.pyplot as plt
from scipy import ndimage

import misc
import camera
import neural_render as nr


def main():
    # mesh = pm.meshutils.generate_icosphere(
        # radius=1, center=np.array([0, 0, 0]))
    mesh = pm.load_mesh(os.path.join(misc.DATA_DIR, 'plate.ply'))
    print(mesh.bbox)
    pts = np.expand_dims(mesh.vertices, 0).astype('float32')
    pts = pts / 10.0
    faces = np.expand_dims(mesh.faces, 0).astype('int32')
    uvs = np.array([[[[0, 0], [0, 1], [1, 0]]]], dtype='float32')
    uvs = np.tile(uvs, reps=tuple(faces.shape[:2]) + (1, 1))
    tex = ndimage.imread(os.path.join(misc.DATA_DIR, 'chessboard.jpg'))
    tex = np.expand_dims(tex, axis=0).astype('float32') / 255.0

    with tf.Session() as session:
        mv = camera.look_at(eye=tf.constant([2, 3, 4], dtype='float32'),
                            center=tf.constant([0, 0, 0], dtype='float32'),
                            up=tf.constant([0, 0, -1], dtype='float32'))
        mv = tf.expand_dims(mv, axis=0)
        H = 3200
        W = 3200
        proj = camera.perspective(focal=3200, H=H, W=W)
        proj = tf.expand_dims(proj, axis=0)

        n = 1
        rendered, uvgrid, z, fids, bc = session.run(
            nr.render(pts=tf.tile(pts, [n, 1, 1]),
                      faces=tf.tile(faces, [n, 1, 1]),
                      uvs=tf.tile(uvs, [n, 1, 1, 1]),
                      tex=tf.tile(tex, [n, 1, 1, 1]),
                      modelview=tf.tile(mv, [n, 1, 1]),
                      proj=tf.tile(proj, [n, 1, 1]),
                      H=H, W=W))
        print('rendering done')

    print(np.min(fids))
    plt.figure()
    plt.imshow(fids[0, :, :])
    plt.show()


if __name__ == '__main__':
    main()