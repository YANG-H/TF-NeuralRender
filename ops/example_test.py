import tensorflow as tf

m = tf.load_op_library('./qt_build_release/libTFNeuralRenderOps.so')

s = tf.Session()
for d in ['/cpu:0', '/gpu:0']:
    with tf.device(d):
        a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype='float32')
    b = m.scaled(a, scale=5)
    print(s.run(b))
