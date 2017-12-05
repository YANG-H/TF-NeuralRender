#include <float.h>
#include <math.h>

#define EIGEN_USE_GPU
#include "neural_render.h"

// is_on_left_side
template <typename T>
XINLINE bool is_on_left_side(const T *p, const T *a, const T *b) {
  T data[9] = {a[0], a[1], 1, b[0], b[1], 1, p[0], p[1], 1};
  T tmp1 = data[0 * 3 + 0] * (data[1 * 3 + 1] * data[2 * 3 + 2] -
                              data[1 * 3 + 2] * data[2 * 3 + 1]);
  T tmp2 = data[0 * 3 + 1] * (data[1 * 3 + 0] * data[2 * 3 + 2] -
                              data[1 * 3 + 2] * data[2 * 3 + 0]);
  T tmp3 = data[0 * 3 + 2] * (data[1 * 3 + 0] * data[2 * 3 + 1] -
                              data[1 * 3 + 1] * data[2 * 3 + 0]);
  return tmp1 - tmp2 + tmp3 >= 0;
}

// is_in_triangle
template <typename T>
XINLINE bool is_in_triangle(const T *p, const T *a, const T *b, const T *c) {
  bool lab = is_on_left_side(p, a, b);
  bool lbc = is_on_left_side(p, b, c);
  bool lca = is_on_left_side(p, c, a);
  return lab == lbc && lbc == lca;
}

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
    /* #bc[0]:
       bx cy - by cx - bx py + by px + cx py - cy px
       ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx

       #bc[1]:
       ax cy - ay cx - ax py + ay px + cx py - cy px
     - ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx

       #bc[2]:
       ax by - ay bx - ax py + ay px + bx py - by px
       ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx
     */
  // clang-format on
  T ax = a[0], ay = a[1];
  T bx = b[0], by = b[1];
  T cx = c[0], cy = c[1];
  T px = p[0], py = p[1];
  T s = ax * by - ay * bx - ax * cy + ay * cx + bx * cy - by * cx;
  if (abs(s) < 1e-6) {
    bc[0] = bc[1] = bc[2] = 1.0 / 3.0;
  } else {
    bc[0] = (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px) / s;
    bc[1] = (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px) / (-s);
    bc[2] = (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px) / s;
  }
}

template <typename T>
XINLINE void add_barycentric_coord_grad(const T *p, const T *a, const T *b,
                                        const T *c, const T *grad_bc, T *grad_a,
                                        T *grad_b, T *grad_c) {
  T ax = a[0], ay = a[1];
  T bx = b[0], by = b[1];
  T cx = c[0], cy = c[1];
  T px = p[0], py = p[1];
  T s = ax * by - ay * bx - ax * cy + ay * cx + bx * cy - by * cx;
  if (abs(s) < 1e-6) {
    return;
  }

  T grad_bc0_ax = -(by - cy) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc0_ay = (bx - cx) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc0_bx = (by - cy) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc0_by = -(bx - cx) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc0_cx = -(by - cy) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);
  T grad_bc0_cy = (bx - cx) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);

  T grad_bc1_ax = (ay - cy) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc1_ay = -(ax - cx) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc1_bx = -(ay - cy) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc1_by = (ax - cx) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc1_cx = (ay - cy) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);
  T grad_bc1_cy = -(ax - cx) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);

  T grad_bc2_ax = -(ay - by) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc2_ay = (ax - bx) * 1.0 / squared(s) *
                  (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px);
  T grad_bc2_bx = (ay - by) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc2_by = -(ax - bx) * 1.0 / squared(s) *
                  (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px);
  T grad_bc2_cx = -(ay - by) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);
  T grad_bc2_cy = (ax - bx) * 1.0 / squared(s) *
                  (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px);

  T grad_ax = grad_bc[0] * grad_bc0_ax + grad_bc[1] * grad_bc1_ax +
              grad_bc[2] * grad_bc2_ax;
  T grad_ay = grad_bc[0] * grad_bc0_ay + grad_bc[1] * grad_bc1_ay +
              grad_bc[2] * grad_bc2_ay;
  T grad_bx = grad_bc[0] * grad_bc0_bx + grad_bc[1] * grad_bc1_bx +
              grad_bc[2] * grad_bc2_bx;
  T grad_by = grad_bc[0] * grad_bc0_by + grad_bc[1] * grad_bc1_by +
              grad_bc[2] * grad_bc2_by;
  T grad_cx = grad_bc[0] * grad_bc0_cx + grad_bc[1] * grad_bc1_cx +
              grad_bc[2] * grad_bc2_cx;
  T grad_cy = grad_bc[0] * grad_bc0_cy + grad_bc[1] * grad_bc1_cy +
              grad_bc[2] * grad_bc2_cy;

  grad_a[0] += grad_ax;
  grad_a[1] += grad_ay;
  grad_b[0] += grad_bx;
  grad_b[1] += grad_by;
  grad_c[0] += grad_cx;
  grad_c[1] += grad_cy;
}

XGLOBAL void
rasterize_kernel(int batch_size, //
                 int npixeliter_each_block, int nfaceiter_each_block,
                 int npoints, int nfaces,
                 const float *pts_data,     // batch_size x npoints x 3
                 const int32_t *faces_data, // batch_size x nfaces x 3
                 const float *uvs_data,     // batch_size x nfaces x 3 x 2
                 int H, int W,
                 float *out_uvgrid_data, // batch_size x H x W x 2
                 float *out_z_data,      // batch_size x H x W
                 int32_t *out_fids_data, // batch_size x H x W
                 float *out_bc_data      // batch_size x H x W x 3
                 ) {

  extern XSHARED float shared_data[];

  // blockDim.xy: [npixels_each_iteration, nfaces_each_iteration]
  int npixels_each_iteration = blockDim.x;
  int pixel_id_this_iteration = threadIdx.x;

  int nfaces_each_iteration = blockDim.y;
  int face_id_this_iteration = threadIdx.y;

  // niterations_each_block
  for (int pixel_iter_id = 0; pixel_iter_id < npixeliter_each_block;
       pixel_iter_id++) {

    /// GET PIXEL INDICES
    /// gridDim.x: [batch_size x (H x W / npixels_each_block)]
    /// ...         [npixeliter_each_block]
    /// blockDim.x: [npixels_each_iteration]
    int global_pixel_id = (blockIdx.x * npixeliter_each_block + pixel_iter_id) *
                              npixels_each_iteration +
                          pixel_id_this_iteration;
    if (global_pixel_id >= batch_size * H * W) { // not a valid pixel here
      continue;
    }
    int pixel_x = global_pixel_id % W;
    int pixel_y = (global_pixel_id / W) % H;
    int batch_id = (global_pixel_id / W / H) % batch_size;

    /// RESULTS ON THIS PIXEL
    float best_z_this_thread = -FLT_MAX;
    int best_fid_this_thread = -1;
    float best_bc_this_thread[3] = {-1.0f, -1.0f, -1.0f};

    for (int face_iter_id = 0; face_iter_id < nfaceiter_each_block;
         face_iter_id++) {

      /// GET FACE INDICES
      /// nfaces: nfaceiter_each_block x nfaces_each_iteration
      /// face_id: face_iter_id, face_id_this_iteration
      int face_id =
          face_iter_id * nfaces_each_iteration + face_id_this_iteration;
      if (face_id >= nfaces) {
        continue;
      }

      /// COMPUTE Z
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
      get_barycentric_coord(pixel_f, ppos[0], ppos[1], ppos[2], bc);

      // get z depth
      if (is_in_triangle(pixel_f, ppos[0], ppos[1], ppos[2])) {
        float z = 0;
        for (int k = 0; k < 3; k++) {
          z += bc[k] * ppos[k][2];
        }
        if (z >= best_z_this_thread) {
          best_z_this_thread = z;
          best_fid_this_thread = face_id;
          for (int k = 0; k < 3; k++) {
            best_bc_this_thread[k] = bc[k];
          }
        }
      }
    }

    // get the shared memory for storing z values on this pixel
    float *z_at_this_pixel =
        shared_data +
        2 * nfaces_each_iteration *
            pixel_id_this_iteration; // [nfaces_each_iteration]
    float *z_for_reduction =
        z_at_this_pixel + nfaces_each_iteration; // [nfaces_each_iteration]
    z_at_this_pixel[face_id_this_iteration] =
        z_for_reduction[face_id_this_iteration] = best_z_this_thread;
    __syncthreads();

    /// FIND THE FACE WITH MAXIMUM Z
    // find the max z and store it to z_for_reduction[0]
    for (unsigned int s = (nfaces_each_iteration + 1) / 2;; s = (s + 1) / 2) {
      if (face_id_this_iteration < s &&
          face_id_this_iteration + s < nfaces_each_iteration) {
        z_for_reduction[face_id_this_iteration] =
            max(z_for_reduction[face_id_this_iteration],
                z_for_reduction[face_id_this_iteration + s]);
      }
      __syncthreads();
      if (s == 1) {
        break;
      }
    }

    /// WRITE RESULTS
    out_z_data[global_pixel_id] = z_for_reduction[0];
    out_uvgrid_data[global_pixel_id * 2 + 0] = -1;
    out_uvgrid_data[global_pixel_id * 2 + 1] = -1;
    out_fids_data[global_pixel_id] = -1;
    __syncthreads();

    if (best_z_this_thread >= z_for_reduction[0]) { // this is the best thread
      out_fids_data[global_pixel_id] = best_fid_this_thread; // write face_id
      if (best_fid_this_thread != -1) {
        for (int k = 0; k < 3; k++) {
          out_bc_data[global_pixel_id * 3 + k] =
              best_bc_this_thread[k]; // write bc
        }

        float puvs[3][2];
        int best_global_face_id = batch_id * nfaces + best_fid_this_thread;
        for (int k = 0; k < 3; k++) {
          puvs[k][0] = uvs_data[(best_global_face_id * 3 + k) * 2 + 0];
          puvs[k][1] = uvs_data[(best_global_face_id * 3 + k) * 2 + 1];
        }
        float u = 0, v = 0;
        for (int k = 0; k < 3; k++) {
          u += best_bc_this_thread[k] * puvs[k][0];
          v += best_bc_this_thread[k] * puvs[k][1];
        }
        // write uvgrid
        out_uvgrid_data[global_pixel_id * 2 + 0] = u;
        out_uvgrid_data[global_pixel_id * 2 + 1] = v;
      }
    }
  }
}

template <>
void RasterizeOp<GPUDevice>::rasterize_impl(
    int batch_size, int npoints, int nfaces, const float *pts_data,
    const int32_t *faces_data, const float *uvs_data, int H, int W,
    float *out_uvgrid_data, float *out_z_data, int32_t *out_fids_data,
    float *out_bc_data) {

  int npixels = batch_size * H * W;

  int nblocks = min(kMaxGridNum, npixels);
  int npixels_each_block = (npixels + nblocks - 1) / nblocks;

  int nfaces_each_iteration = min(kMaxThreadsPerBlock, nfaces);
  int nfaceiter_each_block =
      (nfaces + nfaces_each_iteration - 1) / nfaces_each_iteration;

  int npixels_each_iteration =
      min(kMaxThreadsPerBlock / nfaces_each_iteration, npixels_each_block);
  int npixeliter_each_block =
      (npixels_each_block + npixels_each_iteration - 1) /
      npixels_each_iteration;

  dim3 grid_dim(nblocks);
  CHECK_LE(nblocks, kMaxGridNum);

  const unsigned shared_data_bytes =
      2 * nfaces_each_iteration * npixels_each_iteration * sizeof(float);
  dim3 block_dim(npixels_each_iteration, nfaces_each_iteration);
  CHECK_LE(npixels_each_iteration * nfaces_each_iteration, kMaxThreadsPerBlock);
  CHECK_LE(shared_data_bytes, 64 * 1024);

  XINVOKE_KERNEL(rasterize_kernel, grid_dim, block_dim, shared_data_bytes)
  (batch_size, npixeliter_each_block, nfaceiter_each_block, npoints, nfaces,
   pts_data, faces_data, uvs_data, H, W, out_uvgrid_data, out_z_data,
   out_fids_data, out_bc_data);
}

REGISTER_KERNEL_BUILDER(Name("Rasterize").Device(DEVICE_GPU),
                        RasterizeOp<GPUDevice>)

// a simple version of grad
struct rasterize_direct_grad_kernel {
  XDEVICE void
  operator()(int global_pixel_id, int batch_size, int nfaces, int npoints,
             int H, int W,
             const float *pts_data,         // batch_size x npoints x 3
             const int32_t *faces_data,     // batch_size x nfaces x 3
             const float *uvs_data,         // batch_size x nfaces x 3 x 2
             const int32_t *out_fids_data,  // batch_size x H x W
             const float *out_bc_data,      // batch_size x H x W x 3
             const float *grad_uvgrid_data, // batch_size x H x W x 2
             const float *grad_z_data,      // batch_size x H x W
             float *grad_pts                // batch_size x npoints x 3
             ) const {

    if (global_pixel_id >= batch_size * H * W) { // not a valid pixel here
      return;
    }
    int pixel_x = global_pixel_id % W;
    int pixel_y = (global_pixel_id / W) % H;
    int batch_id = (global_pixel_id / W / H) % batch_size;

    int face_id = out_fids_data[global_pixel_id];
    if (face_id == -1) {
      return;
    }

    // get 3 corner point positions
    int32_t pids[3];
    float ppos[3][3];
    for (int k = 0; k < 3; k++) {
      pids[k] = faces_data[(batch_id * nfaces + face_id) * 3 + k];
      for (int j = 0; j < 3; j++) {
        ppos[k][j] = pts_data[(batch_id * npoints + pids[k]) * 3 + j];
      }
    }

    float puvs[3][2];
    for (int k = 0; k < 3; k++) {
      puvs[k][0] = uvs_data[((batch_id * nfaces + face_id) * 3 + k) * 2 + 0];
      puvs[k][1] = uvs_data[((batch_id * nfaces + face_id) * 3 + k) * 2 + 1];
    }

    const float *bc = out_bc_data + global_pixel_id * 3;
    //  z = 0;
    //  for (int k = 0; k < 3; k++) {
    //    z += bc[k] * ppos[k][2];
    //  }
    const float grad_z = grad_z_data[global_pixel_id];
    //  float u = 0, v = 0;
    //  for (int k = 0; k < 3; k++) {
    //    u += bc[k] * puvs[k][0];
    //    v += bc[k] * puvs[k][1];
    //  }
    const float *grad_uvgrid = grad_uvgrid_data + global_pixel_id * 2;

    // grad_z -> grad_bc
    float grad_bc[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++) {
      grad_bc[k] += grad_z * ppos[k][2];
    }
    // grad_uvgrid -> grad_bc
    for (int k = 0; k < 3; k++) {
      grad_bc[k] += grad_uvgrid[0] * puvs[k][0];
      grad_bc[k] += grad_uvgrid[1] * puvs[k][1];
    }

    float grad_points[3][3];
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        grad_points[k][j] = 0;
      }
    }

    // grad_z -> grad_ppos[k][2]
    for (int k = 0; k < 3; k++) {
      grad_points[k][2] += grad_z * bc[k];
    }

    float pixel_fx = (pixel_x + 0.5f) / W;
    pixel_fx = pixel_fx * 2 - 1; // [-1, 1]
    float pixel_fy = (pixel_y + 0.5f) / H;
    pixel_fy = pixel_fy * 2 - 1; // [-1, 1]
    float pixel_f[2] = {pixel_fx, pixel_fy};
    // grad_bc -> grad_ppos[k][0, 1]
    add_barycentric_coord_grad(pixel_f, ppos[0], ppos[1], ppos[2], grad_bc,
                               grad_points[0], grad_points[1], grad_points[2]);

    // write results
    for (int k = 0; k < 3; k++) {
      int global_point_id = batch_id * npoints + pids[k];
      for (int j = 0; j < 3; j++) {
        atomicAdd(grad_pts + global_point_id * 3 + j, grad_points[k][j]);
      }
    }
  }
};

template <>
void RasterizeGradOp<GPUDevice>::rasterize_grad_impl(
    int batch_size, int nfaces, int npoints, int H, int W,
    const float *pts_data, const int32_t *faces_data, const float *uvs_data,
    const int32_t *out_fids_data, const float *out_bc_data,
    const float *grad_uvgrid_data, const float *grad_z_data,
    float *grad_pts_data) {
  Kernel<GPUDevice>::Launch(rasterize_direct_grad_kernel(), batch_size * H * W,
                            batch_size, nfaces, npoints, H, W, pts_data,
                            faces_data, uvs_data, out_fids_data, out_bc_data,
                            grad_uvgrid_data, grad_z_data, grad_pts_data);
}

REGISTER_KERNEL_BUILDER(Name("RasterizeGrad").Device(DEVICE_GPU),
                        RasterizeGradOp<GPUDevice>)

struct bilinear_sample_kernel {
  XINLINE void operator()(int idx, // BxHxW
                          int batch_size, int Ht, int Wt, int Dt, int H, int W,
                          const float *tex_data, // BxHtxWtxDt
                          const float *uvgrid_data, float *out_data) const {
    int batch_id = idx / W / H % batch_size;

    float u = uvgrid_data[idx * 2 + 0];
    float v = uvgrid_data[idx * 2 + 1];
    float uu_real = u * (Wt - 1) + 0.5;
    float vv_real = v * (Ht - 1) + 0.5;
    int uu = static_cast<int>(floorf(uu_real));
    int vv = static_cast<int>(floorf(vv_real));

    int uu2 = uu + 1;
    int vv2 = vv + 1;

    float uu_w = uu_real - uu;
    float vv_w = vv_real - vv;

    uu = min(Wt - 1, max(0, uu));
    vv = min(Ht - 1, max(0, vv));
    uu2 = min(Wt - 1, max(0, uu2));
    vv2 = min(Ht - 1, max(0, vv2));

    int tid_topleft = (batch_id * Ht + vv) * Wt + uu;
    int tid_bottomright = (batch_id * Ht + vv2) * Wt + uu2;
    int tid_topright = (batch_id * Ht + vv) * Wt + uu2;
    int tid_bottomleft = (batch_id * Ht + vv2) * Wt + uu;
    for (int k = 0; k < Dt; k++) {
      float color = tex_data[tid_topleft * Dt + k] * (1 - uu_w) * (1 - vv_w) +
                    tex_data[tid_topright * Dt + k] * uu_w * (1 - vv_w) +
                    tex_data[tid_bottomleft * Dt + k] * (1 - uu_w) * vv_w +
                    tex_data[tid_bottomright * Dt + k] * uu_w * vv_w;
      out_data[idx * Dt + k] = color;
    }
  }
};

template <>
void BilinearSampleOp<GPUDevice>::impl(int batch_size, int Ht, int Wt, int Dt,
                                       int H, int W, const float *tex_data,
                                       const float *uvgrid_data,
                                       float *out_data) {
  Kernel<GPUDevice>::Launch(bilinear_sample_kernel(), batch_size * H * W,
                            batch_size, Ht, Wt, Dt, H, W, tex_data, uvgrid_data,
                            out_data);
}

REGISTER_KERNEL_BUILDER(Name("BilinearSample").Device(DEVICE_GPU),
                        BilinearSampleOp<GPUDevice>)
