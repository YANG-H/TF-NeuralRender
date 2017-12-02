import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL = tf.load_op_library(os.path.join(
    BASE_DIR, 'qt_build_release', 'libTFNeuralRenderOps.so'))


def scaled(data, scale):
    return MODEL.scaled(data, scale=scale)

s = tf.Session()
for d in ['/cpu:0', '/gpu:0']:
    with tf.device(d):
        a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype='float32')
    b = scaled(a, scale=5)
    print(s.run(b))
