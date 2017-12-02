import os
import sys
import numpy as np
import tensorflow as tf
import neural_render as nr
import spatial_transform as st
import pymesh as pm
import matplotlib.pyplot as plt
from scipy import ndimage

import camera
import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def main():
    # mesh = pm.load_mesh(os.path.join(BASE_DIR, 'data', 'cube.obj'))
    mesh = pm.meshutils.generate_icosphere(
        radius=1, center=np.array([0, 0, 0]))
    uvs = np.array([[[0, 0], [0, 1], [1, 0]]], dtype='float32')
    uvs = np.tile(uvs, reps=[mesh.faces.shape[0], 1, 1])

    H = 800
    W = 800
    focal = 800
    modelview = camera.look_at(eye=np.array([2, 3, 3.2]),
                               center=np.array([0, 0, 0]),
                               up=np.array([0, 0, -1]))
    proj = camera.perspective(focal=focal, H=H, W=W)
    pts = camera.apply_transform(mesh.vertices, modelview, proj)

    pts = np.expand_dims(pts, axis=0)
    faces = np.expand_dims(mesh.faces, axis=0)
    uvs = np.expand_dims(uvs, axis=0)

    pts_v = tf.constant(pts.astype('float32'))
    faces_v = tf.constant(faces.astype('int32'))
    uvs_v = tf.constant(uvs.astype('float32'))

    uvgrid_v, z_v, fids_v, bc_v = nr.rasterize(
        pts=pts_v, faces=faces_v, uvs=uvs_v, H=H, W=W)

    with tf.Session() as ss:
        uvgrid, fids, z, bc = ss.run([uvgrid_v, fids_v, z_v, bc_v])
        print(uvgrid)
        print(np.max(fids))
        print(np.max(z))

        z[np.where(z < 0)] = np.min(z[np.where(z >= 0)])
        plt.figure()
        plt.imshow(z[0, :, :])
        # plt.imshow(uvgrid[0, :, :, 0])
        plt.show()


if __name__ == '__main__':
    main()
