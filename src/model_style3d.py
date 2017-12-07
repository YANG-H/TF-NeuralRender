import os
import sys
import time
import math
import functools
import shutil
import random
import tensorflow as tf
import numpy as np
import pymesh as pm
from scipy import ndimage

import neural_render as nr
import camera
import vgg

import misc
from mesh import gen_uv_texshape

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


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
             learning_rate=1e-3, tex_unit_size=10,
             subsample=4, try_resuming=True):

    assert len(content_targets_pts.shape) == 2
    assert len(faces.shape) == 2
    print('npoints=%d, nfaces=%d' %
          (content_targets_pts.shape[0], faces.shape[0]))

    uvs, tex_shape = gen_uv_texshape(faces.shape[0], tex_unit_size)

    batch_size = ncams

    style_features = {}

    style_shape = (1,) + style_target_img.shape
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/gpu:0'), tf.Session() as sess:
        style_image = tf.constant(np.array([style_target_img]),
                                  dtype=tf.float32)
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        for layer in STYLE_LAYERS:
            features = sess.run(net[layer])
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.device('/gpu:0'):
            X_pts = tf.constant(value=content_targets_pts, dtype=tf.float32)
            pred_pts = tf.Variable(
                initial_value=content_targets_pts, trainable=True,
                dtype=tf.float32)
            pred_tex = tf.Variable(
                initial_value=tf.random_normal(
                    (tex_shape[0], tex_shape[1], 3)) * 0.256,
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
            uvs_batched = tf.tile(tf.expand_dims(
                uvs, axis=0), multiples=[batch_size, 1, 1, 1])

            # make view matrix
            view_angles = tf.random_uniform(
                [ncams, 1], minval=0, maxval=1, dtype=tf.float32)
            view_dists = tf.random_uniform(
                [ncams, 1], minval=1, maxval=3, dtype=tf.float32)
            view_hs = tf.random_uniform(
                [ncams, 1], minval=-1, maxval=5, dtype=tf.float32)
            modelviews = camera.batched_look_at(
                eye=tf.concat([tf.cos(view_angles * math.pi * 2) * view_dists,
                               tf.sin(view_angles * math.pi * 2) * view_dists,
                               view_hs], axis=1),
                center=tf.zeros([ncams, 3], dtype=tf.float32),
                up=tf.tile(tf.constant([[0, 0, 1]], dtype=tf.float32),
                           [ncams, 1]))

            projs = tf.tile(tf.expand_dims(
                camera.perspective(focal=200, H=255, W=255), axis=0),
                [ncams, 1, 1])

            pred_rendered, uvgrid, z, fids, bc = nr.render(
                pts=pred_pts_batched, faces=faces_batched, uvs=uvs_batched,
                tex=pred_tex_batched, modelview=modelviews, proj=projs,
                H=255 * subsample, W=255 * subsample)
            pooled_pred_rendered = tf.nn.avg_pool(
                pred_rendered, ksize=[1, subsample, subsample, 1],
                strides=[1, subsample, subsample, 1], padding='VALID')

            net = vgg.net(vgg_path, pooled_pred_rendered)

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

            # total variation denoising
            tv_y_size = _tensor_size(pred_tex[1:, :, :])
            tv_x_size = _tensor_size(pred_tex[:, 1:, :])
            y_tv = tf.nn.l2_loss(pred_tex[1:, :, :] -
                                 pred_tex[:pred_tex.shape[0] - 1, :, :])
            x_tv = tf.nn.l2_loss(pred_tex[:, 1:, :] -
                                 pred_tex[:, :pred_tex.shape[1] - 1, :])
            tv_loss = tv_weight * 2 * \
                (x_tv / tv_x_size + y_tv / tv_y_size)

            loss = content_loss + style_loss + tv_loss

        # summary
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("content_loss", content_loss)
        tf.summary.scalar("style_loss", style_loss)
        tf.summary.scalar("tv_loss", tv_loss)
        tf.summary.image("rendered", pred_rendered)
        tf.summary.image("pooled_rendered", pooled_pred_rendered)
        tf.summary.image("tex", tf.expand_dims(pred_tex, axis=0))
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir)
        summary_writer.add_graph(tf.get_default_graph())

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        # resume session if permitted
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if try_resuming and ckpt and ckpt.model_checkpoint_path:
            print('session restored from %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(epochs):
            start_time = time.time()
            if epoch % 4 == 0:
                _, l, cl, sl, tl, summary = sess.run(
                    [train_step, loss, content_loss, style_loss, tv_loss,
                     merged_summary_op])
                summary_writer.add_summary(summary, epoch)
            else:
                _, l, cl, sl, tl = sess.run(
                    [train_step, loss, content_loss, style_loss, tv_loss])
            end_time = time.time()
            delta_time = end_time - start_time
            print('%d: loss=%f, content_loss=%f, style_loss=%f, tv_loss=%f, '
                  'time cost=%f seconds' % (
                      epoch, l, cl, sl, tl, delta_time))
            if epoch % 100 == 0:
                saver.save(sess, os.path.join(
                    log_dir, 'model'), global_step=epoch)


def main():
    mesh = pm.load_mesh(os.path.join(misc.DATA_DIR, 'mesh', 'bunny.obj'))
    bmin, bmax = mesh.bbox
    pts = mesh.vertices
    pts = pts - np.tile(np.expand_dims((bmax + bmin) / 2,
                                       axis=0), [mesh.num_vertices, 1])
    pts = pts / np.max(np.linalg.norm(pts, axis=1))
    pts = pts[:, [0, 2, 1]]
    faces = mesh.faces

    style_img = ndimage.imread(os.path.join(misc.DATA_DIR, 'style',
                                            'la_muse.jpg'), mode='RGB')

    optimize(pts, faces, 8, style_img,
             content_weight=1e7, style_weight=1e-1, tv_weight=0,
             epochs=50000, learning_rate=1e-1, tex_unit_size=4,
             subsample=4,
             log_dir=os.path.join(misc.BASE_DIR, 'log', '1'))


if __name__ == '__main__':
    main()
