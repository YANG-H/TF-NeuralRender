#include <float.h>
#include <math.h>

#define EIGEN_USE_GPU
#include "neural_render.h"

template <typename T>
XINLINE bool between(T value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

// bc[0]*(b-a) + bc[1]*(c-a) = p-a
// p = (1-bc[0]-bc[1])*a + bc[0]*b + bc[1]*c
template <typename T>
XINLINE void get_barycentric_coord(const T *p, const T *a, const T *b,
                                   const T *c, T *bc) {
  // clang-format off
  //    / (ay - cy) (ax - px)   (ax - cx) (ay - py)  (ax - bx) (ay - py)   (ay - by) (ax - px) |
  //    | ------------------- - -------------------, ------------------- - ------------------- |
  //    \          #1                    #1                   #1                    #1         /
  //    where
  //       #1 == ax by - ay bx - ax cy + ay cx + bx cy - by cx
  // clang-format on
  T s = a[0] * b[1] - a[1] * b[0] - a[0] * c[1] + a[1] * c[0] + b[0] * c[1] -
        b[1] * c[0];
  bc[0] = (a[1] - c[1]) * (a[0] - p[0]) - (a[0] - c[0]) * (a[1] - p[1]);
  bc[0] /= s;
  bc[1] = (a[0] - b[0]) * (a[1] - p[1]) - (a[1] - b[1]) * (a[0] - p[0]);
  bc[1] /= s;
}

__global__ void
rasterize_kernel(int batch_size, int npoints,
                 const float *pts_data,     // batch_size x npoints x 3
                 const int32_t *faces_data, // batch_size x nfaces x 3
                 const float *uvs_data,     // batch_size x nfaces x 3 x 2
                 int H, int W,
                 float *out_uvgrid_data, // batch_size x H x W x 2
                 int32_t *out_fids_data, // batch_size x H x W
                 float *out_z_data       // batch_size x H x W
                 ) {

  extern XSHARED float shared_data[];
  float *z_at_this_pixel = shared_data;                  // [nfaces]
  float *z_for_reduction = z_at_this_pixel + blockDim.x; // [nfaces]

  int nfaces = blockDim.x;

  // blockDim.x: [nfaces]
  int face_id = threadIdx.x;

  // gridDim.x: [batch_size x H x W]
  int pixel_x = blockIdx.x % W;
  int pixel_y = (blockIdx.x / W) % H;
  int batch_id = (blockIdx.x / W / H) % batch_size;

  /// debuging
  //  int ii = (batch_id * H + pixel_y) * W + pixel_x;
  //  out_z_data[ii] = -1;
  //  out_fids_data[ii] = -1;
  //  out_uvgrid_data[ii * 2] = -1;
  //  out_uvgrid_data[ii * 2 + 1] = -1;
  //  return;

  // compute z and store it to z_at_this_pixel and z_for_reduction
  // get 3 corner point positions
  int32_t pids[3];
  float ppos[3][3];
  for (int k = 0; k < 3; k++) {
    pids[k] = faces_data[(batch_id * nfaces + face_id) * 3 + k];
    for (int j = 0; j < 3; j++) {
      ppos[k][j] = pts_data[(batch_id * npoints + pids[k]) * 3 + j];
    }
  }
  // compute barycentric coords
  float pixel_fx = (pixel_x + 0.5f) / W;
  pixel_fx = pixel_fx * 2 - 1; // [-1, 1]
  float pixel_fy = (pixel_y + 0.5f) / H;
  pixel_fy = pixel_fy * 2 - 1; // [-1, 1]
  float pixel_f[2] = {pixel_fx, pixel_fy};
  float bc[3];
  // p = (1-bc[1]-bc[2])*a + bc[1]*b + bc[2]*c
  get_barycentric_coord(pixel_f, ppos[0], ppos[1], ppos[2], bc + 1);
  bc[0] = 1 - bc[1] - bc[2];

  bool inside_face = true;
  for (int k = 0; k < 3; k++) {
    if (bc[k] < 0 || bc[k] > 1) {
      inside_face = false;
      break;
    }
  }

  // get z depth
  float z = -FLT_MAX;
  if (inside_face) {
    z = 0;
    for (int k = 0; k < 3; k++) {
      z += bc[k] * ppos[k][2];
    }
  }
  z_at_this_pixel[face_id] = z_for_reduction[face_id] = z;
  __syncthreads();

  // find the max z and store it to z_for_reduction[0]
  for (unsigned int s = (blockDim.x + 1) / 2; s > 1; s = (s + 1) / 2) {
    if (face_id < s && face_id + s < blockDim.x) {
      z_for_reduction[face_id] =
          max(z_for_reduction[face_id], z_for_reduction[face_id + s]);
    }
    __syncthreads();
  }

  // write max z
  int out_loc = (batch_id * H + pixel_y) * W + pixel_x;
  out_z_data[out_loc] = z_for_reduction[0];
  out_fids_data[out_loc] = -1;
  out_uvgrid_data[out_loc * 2 + 0] = -1;
  out_uvgrid_data[out_loc * 2 + 1] = -1;

  // write face_id and uvgrid
  if (inside_face &&
      z_at_this_pixel[face_id] ==
          z_for_reduction[0]) { // this is the visible face_id

    out_fids_data[out_loc] = face_id;

    float puvs[3][2];
    for (int k = 0; k < 3; k++) {
      puvs[k][0] = uvs_data[((batch_id * nfaces + face_id) * 3 + k) * 2 + 0];
      puvs[k][1] = uvs_data[((batch_id * nfaces + face_id) * 3 + k) * 2 + 1];
    }
    float u = 0, v = 0;
    for (int k = 0; k < 3; k++) {
      u += bc[k] * puvs[k][0];
      v += bc[k] * puvs[k][1];
    }
    out_uvgrid_data[out_loc * 2 + 0] = u;
    out_uvgrid_data[out_loc * 2 + 1] = v;
  }
}

void rasterize_impl(GPUDevice, int batch_size, int npoints, int nfaces,
                    const float *pts_data, const int32_t *faces_data,
                    const float *uvs_data, int H, int W, float *out_uvgrid_data,
                    int32_t *out_fids_data, float *out_z_data) {
  const unsigned shared_data_bytes = 2 * nfaces * sizeof(float);
  int grid_dim = batch_size * H * W;
  // todo: grid_dim is likely to be larger than kMaxGridNum, fix this!

  int block_dim = nfaces;
  XINVOKE_KERNEL(rasterize_kernel, grid_dim, block_dim, shared_data_bytes)
  (batch_size, npoints, pts_data, faces_data, uvs_data, H, W, out_uvgrid_data,
   out_fids_data, out_z_data);
}

REGISTER_KERNEL_BUILDER(Name("Rasterize").Device(DEVICE_GPU),
                        RasterizeOp<GPUDevice>)
