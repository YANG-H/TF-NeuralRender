import os
import sys
import time
import math
import functools
import tensorflow as tf
import numpy as np
import pymesh as pm
from scipy import ndimage

import neural_render as nr
import camera
import vgg

import misc

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'


def _make_uvs(nfaces):
    uvs = tf.stack([tf.range(1, nfaces + 1, dtype=tf.float32) / (nfaces),
                    tf.zeros([nfaces], dtype=tf.float32),
                    tf.range(1, nfaces + 1, dtype=tf.float32) / (nfaces),
                    tf.ones([nfaces], dtype=tf.float32),
                    tf.range(0, nfaces, dtype=tf.float32) / (nfaces),
                    tf.ones([nfaces], dtype=tf.float32)])
    uvs = tf.transpose(uvs, [1, 0])  # nfaces x 6
    uvs = tf.reshape(uvs, [nfaces, 3, 2])  # nfaces x 3 x 2
    return uvs


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def optimize(content_targets_pts, faces,  # single mesh
             ncams,  # number of cams
             style_target_img,  # single image
             content_weight, style_weight, tv_weight, epochs=10,
             vgg_path=os.path.join(misc.DATA_DIR, 'VGG',
                                   'imagenet-vgg-verydeep-19.mat'),
             log_dir=os.path.join(misc.BASE_DIR, 'log'),
             learning_rate=1e-3):

    assert len(content_targets_pts.shape) == 2
    assert len(faces.shape) == 2

    batch_size = ncams

    style_features = {}

    style_shape = (1,) + style_target_img.shape
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(
            tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target_img])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_pts = tf.placeholder(
            tf.float32, shape=content_targets_pts.shape, name='X_pts')

        pred_pts = tf.Variable(
            initial_value=content_targets_pts, trainable=True,
            dtype=tf.float32)
        pred_tex = tf.Variable(
            initial_value=tf.random_normal((5, 5 * faces.shape[0], 3)) * 0.256,
            trainable=True, dtype=tf.float32)

        # content(shape) loss
        npoints = content_targets_pts.shape[0]
        content_loss = content_weight * \
            (tf.nn.l2_loss(X_pts - pred_pts) / npoints)

        # style losses
        pred_pts_batched = tf.tile(tf.expand_dims(pred_pts, axis=0),
                                   multiples=[batch_size, 1, 1])  # BxNpx3
        pred_tex_batched = tf.tile(tf.expand_dims(pred_tex, axis=0),
                                   multiples=[batch_size, 1, 1, 1])
        faces_batched = tf.tile(tf.expand_dims(
            faces, axis=0), multiples=[batch_size, 1, 1])
        uvs = _make_uvs(nfaces=faces.shape[0])
        uvs_batched = tf.tile(tf.expand_dims(
            uvs, axis=0), multiples=[batch_size, 1, 1, 1])

        def _make_modelview(i):
            return camera.look_at(
                eye=[math.cos(i / ncams * math.pi * 2),
                     math.sin(i / ncams * math.pi * 2), 1],
                center=[0, 0, 0], up=[0, 0, 1])
        modelviews = tf.stack([_make_modelview(i) for i in range(ncams)])
        projs = tf.tile(tf.expand_dims(camera.perspective(focal=300, H=255, W=255), axis=0),
                        [ncams, 1, 1])

        pred_rendered, uvgrid, z, fids, bc = nr.render(
            pts=pred_pts_batched, faces=faces_batched, uvs=uvs_batched,
            tex=pred_tex_batched, modelview=modelviews, proj=projs,
            H=255, W=255)  # batch_size x 255 x 255 x 3

        net = vgg.net(vgg_path, pred_rendered)

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(
                lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(
                2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
        style_loss = style_weight * \
            functools.reduce(tf.add, style_losses) / batch_size

        # # total variation denoising
        # tv_y_size = _tensor_size(preds[:, 1:, :, :])
        # tv_x_size = _tensor_size(preds[:, :, 1:, :])
        # y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] -
        #                      preds[:, :batch_shape[1] - 1, :, :])
        # x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] -
        #                      preds[:, :, :batch_shape[2] - 1, :])
        # tv_loss = tv_weight * 2 * \
        #     (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        loss = content_loss + style_loss

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("content_loss", content_loss)
        tf.summary.scalar("style_loss", style_loss)
        tf.summary.image("rendered", pred_rendered)        
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            log_dir, graph=tf.get_default_graph())
        summary_writer.add_graph(tf.get_default_graph())

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            start_time = time.time()
            _, summary = sess.run([train_step, merged_summary_op], feed_dict={
                                  X_pts: content_targets_pts})
            end_time = time.time()
            delta_time = end_time - start_time
            print('%d: time cost=%f seconds' % (epoch, delta_time))
            summary_writer.add_summary(summary, epoch)


def main():
    mesh = pm.load_mesh(os.path.join(misc.DATA_DIR, 'cube.obj'))
    # mesh = pm.meshutils.generate_icosphere(radius=1, center=np.zeros([3]))
    print(mesh.bbox)
    bmin, bmax = mesh.bbox
    pts = mesh.vertices
    pts = pts - np.tile(np.expand_dims((bmax + bmin) / 2,
                                       axis=0), [mesh.num_vertices, 1])
    pts = pts / np.max(np.linalg.norm(pts, axis=1))
    faces = mesh.faces

    style_img = ndimage.imread(os.path.join(misc.DATA_DIR, 'style', 'wave.jpg'),
                               mode='RGB')

    optimize(pts, faces, 4, style_img, 
             content_weight=10, style_weight=100, tv_weight=0)


if __name__ == '__main__':
    main()
