import tensorflow as tf
import numpy as np


def perspective(focal, H, W, z_near=1e-2, z_far=1e2):
    ''' make a single projection matrix '''
    aa = -(z_far + z_near) / (z_far - z_near)
    bb = -2 * z_far * z_near / (z_far - z_near)
    # [2f/W     0       -2px/W+1    0 ]
    # [0        2f/H    -2py/H+1    0 ]
    # [0        0       aa          bb]
    # [0        0       -1          0 ]
    return tf.cast(tf.stack([[2 * focal / W, 0, 0, 0],
                             [0, 2 * focal / H, 0, 0],
                             [0, 0,  aa, bb],
                             [0, 0, -1, 0]]), tf.float32)


def look_at(eye, center, up):
    ''' make a single view matrix '''
    def _normalize(xx):
        return xx / tf.norm(xx)

    z = _normalize(center - eye)
    x = _normalize(tf.cross(up, z))
    y = tf.cross(z, x)

    # [-x0 -x1 -x2 a]
    # [ y0  y1  y2 b]
    # [-z0 -z1 -z2 c]
    # [  0   0   0 1]

    def _dot(xx, yy):
        return tf.reduce_sum(xx * yy)
    a = _dot(x, eye)
    b = -_dot(y, eye)
    c = _dot(z, eye)
    return tf.cast(tf.stack([[-x[0], -x[1], -x[2], a],
                             [y[0], y[1], y[2], b],
                             [-z[0], -z[1], -z[2], c],
                             [0, 0, 0, 1]]), tf.float32)


def apply_transform(pts, modelview, proj):
    assert len(pts.shape) == 3
    pts = tf.concat((pts, tf.ones(tuple(pts.shape[:-1]) + (1,))), axis=-1)
    pts = tf.matmul(pts, tf.transpose(tf.matmul(proj, modelview), [0, 2, 1]))
    return pts[:, :, 0:3] / tf.expand_dims(pts[:, :, 3], axis=-1)
