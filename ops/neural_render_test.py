import os
import sys
import numpy as np
import tensorflow as tf
import neural_render as nr
import pymesh as pm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def perspective(focal, H, W, z_near=1e-2, z_far=1e2, pp=None):
    if pp is None:
        pp = np.array([W/2, H/2])
    result = np.zeros([4, 4], dtype=float)
    result[0, 0] = 2 * focal / W
    result[1, 1] = 2 * focal / H
    result[0, 2] = - (2 * pp[0] / W - 1)
    result[1, 2] = - (2 * pp[1] / H - 1)
    result[2, 2] = -(z_far + z_near) / (z_far - z_near)
    result[2, 3] = -2 * z_far*z_near / (z_far - z_near)
    result[3, 2] = -1
    # result[3, 3] = 1
    return result


def normalize(v):
    return v / np.linalg.norm(v)


def look_at(eye, center, up):
    z = normalize(center - eye)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)

    result = np.zeros([4, 4], dtype=float)
    result[0, 0] = -x[0]
    result[0, 1] = -x[1]
    result[0, 2] = -x[2]

    result[1, 0] = y[0]
    result[1, 1] = y[1]
    result[1, 2] = y[2]

    result[2, 0] = -z[0]
    result[2, 1] = -z[1]
    result[2, 2] = -z[2]

    result[0, 3] = np.dot(x, eye)
    result[1, 3] = -np.dot(y, eye)
    result[2, 3] = np.dot(z, eye)
    result[3, 3] = 1
    return result


def transform(mat, pts, out4d=False):
    pts = np.concatenate((pts, np.ones([pts.shape[0], 1])), axis=1)
    pts = np.matmul(pts, np.transpose(mat))
    return pts if out4d else pts[:, 0:3] / np.expand_dims(pts[:, 3], axis=-1)


def main():
    mesh = pm.load_mesh(os.path.join(BASE_DIR, 'data', 'cube.obj'))
    uvs = np.array([[[0, 0], [0, 1], [1, 0]]], dtype='float32')
    uvs = np.tile(uvs, reps=[mesh.faces.shape[0], 1, 1])

    H = 400
    W = 400
    mv = look_at(eye=np.array([2, 3, 4]), center=np.array(
        [0, 0, 0]), up=np.array([0, 0, 1]))
    proj = perspective(focal=200, H=H, W=W)

    pts = mesh.vertices
    print(np.max(pts, axis=0))
    print(np.min(pts, axis=0))

    pts = transform((proj * mv), pts, out4d=True)
    print(np.max(pts, axis=0))
    print(np.min(pts, axis=0))
    pts = pts[:, 0:3]

    pts = np.expand_dims(pts, axis=0)
    faces = np.expand_dims(mesh.faces, axis=0)
    uvs = np.expand_dims(uvs, axis=0)

    pts_v = tf.constant(pts.astype('float32'))
    faces_v = tf.constant(faces.astype('int32'))
    uvs_v = tf.constant(uvs.astype('float32'))
    uvgrid_v, fids_v, z_v = nr.rasterize(pts=pts_v, faces=faces_v, uvs=uvs_v, H=H, W=W)
    with tf.Session() as ss:
        uvgrid, fids, z = ss.run([uvgrid_v, fids_v, z_v])
        print(uvgrid.shape)
        print(fids.shape)
        print(z.shape)
        print(np.min(fids))

    


if __name__ == '__main__':
    main()

