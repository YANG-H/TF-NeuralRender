import os
import sys
import math
import pymesh as pm
import numpy as np

import misc


def gen_fake_uv(nfaces):
    return np.tile([[[0, 1], [1, 1], [1, 0]]], [
        nfaces, 1, 1]).astype('float32')


def gen_uv_texshape(nfaces, tex_size_each_face=8):
    m = int(math.ceil(math.sqrt(nfaces)))
    n = (nfaces + m - 1) // m
    assert m * n >= nfaces
    uvs = np.zeros([nfaces, 3, 2], dtype='float32')
    for i in range(nfaces):
        x = i % n  # 0, n-1
        y = i // n  # 0, m-1
        uvs[i, 0, :] = [x + 1, y]
        uvs[i, 1, :] = [x + 1, y + 1]
        uvs[i, 2, :] = [x, y + 1]
    uvs[:, :, 0] /= n
    uvs[:, :, 1] /= m
    return uvs, (m * tex_size_each_face, n * tex_size_each_face)


def load_obj(fname, normalize_pts=False):
    def _parse_f(ids):
        fd = [(int(x) if x != '' else -1) for x in ids.split('/')]
        fd += [-1] * (3 - len(fd))
        return fd
    with open(fname, 'r') as f:
        vdata = []
        vtdata = []
        fdata = []
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if tokens[0] == '#':
                continue
            if tokens[0] == 'v':
                vdata += [[float(x) for x in tokens[1:]]]
                continue
            if tokens[0] == 'vt':
                vtdata += [[float(x) for x in tokens[1:]]]
                continue
            if tokens[0] == 'f':
                fdata += [[_parse_f(b) for b in tokens[1:]]]
        vdata = np.array(vdata)
        vtdata = np.array(vtdata)
        fdata = np.array(fdata)
        nfaces = fdata.shape[0]
        uvdata = np.ones([nfaces, 3, 2]) * -1
        valid_uv_inds = np.where(fdata[:, 0, 1] != -1)[0]
        for i in range(3):
            uvdata[valid_uv_inds, i, :] = (vtdata[
                fdata[valid_uv_inds, i, 1] - 1, :] + 1) / 2
        fdata = fdata[:, :, 0] - 1

        pts = vdata
        if normalize_pts:
            bmin, bmax = np.min(pts, axis=0), np.max(pts, axis=0)
            pts = pts - np.tile(np.expand_dims((bmax + bmin) / 2,
                                               axis=0), [pts.shape[0], 1])
            pts = pts / np.max(np.linalg.norm(pts, axis=1))
            pts = pts[:, [0, 2, 1]]

        return pts, fdata, uvdata


def main():
    load_obj(os.path.join(misc.DATA_DIR, 'chair.obj'))


if __name__ == '__main__':
    main()
