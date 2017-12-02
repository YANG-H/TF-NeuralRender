#pragma once

#include "common.h"

// out_uvgrid, out_z, out_fids, out_bc =
//     rasterize(pts, faces, uvs, screen_shape)
template <typename Device> class RasterizeOp : public OpKernel {
  static void rasterize_impl(int batch_size, int npoints, int nfaces,
                             const float *pts_data, const int32_t *faces_data,
                             const float *uvs_data, int H, int W,
                             float *out_uvgrid_data, float *out_z_data,
                             int32_t *out_fids_data, float *out_bc_data);

public:
  explicit RasterizeOp(OpKernelConstruction *context) : OpKernel(context) {
    int32_t H, W;
    context->GetAttr("H", &H);
    context->GetAttr("W", &W);
    screen_shape_ = TensorShape{H, W};
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &pts = context->input(0); // BxNpx3, float, device coord
    auto pts_data = pts.flat<float>().data();
    const Tensor &faces = context->input(1); // BxNfx3, int
    auto faces_data = faces.flat<int32_t>().data();
    const Tensor &uvs = context->input(2); // BxNfx3x2, float
    auto uvs_data = uvs.flat<float>().data();

    int batch_size = pts.dim_size(0);
    OP_REQUIRES(context, batch_size == faces.dim_size(0),
                errors::InvalidArgument("batch sizes mismatch"));
    OP_REQUIRES(context, batch_size == uvs.dim_size(0),
                errors::InvalidArgument("batch sizes mismatch"));

    int np = pts.dim_size(1);
    int nf = faces.dim_size(1);

    Tensor *out_uvgrid_ptr = nullptr; // BxHxWx2
    Tensor *out_z_ptr = nullptr;      // BxHxW
    Tensor *out_fids_ptr = nullptr;   // BxHxW
    Tensor *out_bc_ptr = nullptr;     // BxHxWx3
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1), 2},
                       &out_uvgrid_ptr));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1)},
                       &out_z_ptr));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       2, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1)},
                       &out_fids_ptr));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       3, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1), 3},
                       &out_bc_ptr));
    auto out_uvgrid_data = out_uvgrid_ptr->flat<float>().data();
    auto out_z_data = out_z_ptr->flat<float>().data();
    auto out_fids_data = out_fids_ptr->flat<int32_t>().data();
    auto out_bc_data = out_bc_ptr->flat<float>().data();

    int H = screen_shape_.dim_size(0);
    int W = screen_shape_.dim_size(1);
    rasterize_impl(batch_size, np, nf, pts_data, faces_data, uvs_data, H, W,
                   out_uvgrid_data, out_z_data, out_fids_data, out_bc_data);
  }

private:
  TensorShape screen_shape_;
};

// grad_pts = rasterize_grad(pts, faces, uvs, out_uvgrid, out_z, out_fids,
//      out_bc, grad_uvgrid, grad_z)
template <typename Device> class RasterizeGradOp : public OpKernel {
  static void
  rasterize_grad_impl(int batch_size, int nfaces, int npoints, int H, int W,
                      const float *pts_data, const int32_t *faces_data,
                      const float *uvs_data, const float *out_uvgrid_data,
                      const float *out_z_data, const int32_t *out_fids_data,
                      const float *out_bc_data, const float *grad_uvgrid_data,
                      const float *grad_z_data, float *grad_pts);

public:
  explicit RasterizeGradOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &pts = context->input(0); // BxNpx3, float, device coord
    auto pts_data = pts.flat<float>().data();
    const Tensor &faces = context->input(1); // BxNfx3, int
    auto faces_data = faces.flat<int32_t>().data();
    const Tensor &uvs = context->input(2); // BxNfx3x2, float
    auto uvs_data = uvs.flat<float>().data();

    const Tensor &out_uvgrid = context->input(3); // BxHxWx2, float
    auto out_uvgrid_data = out_uvgrid.flat<float>().data();
    const Tensor &out_z = context->input(4); // BxHxW, float
    auto out_z_data = out_z.flat<float>().data();
    const Tensor &out_fids = context->input(5); // BxHxW, int
    auto out_fids_data = out_fids.flat<int32_t>().data();
    const Tensor &out_bc = context->input(6); // BxHxWx3, float
    auto out_bc_data = out_bc.flat<float>().data();

    const Tensor &grad_uvgrid = context->input(7); // BxHxWx2, float
    auto grad_uvgrid_data = grad_uvgrid.flat<float>().data();
    const Tensor &grad_z = context->input(8); // BxHxW, float
    auto grad_z_data = grad_z.flat<float>().data();

    int batch_size = pts.dim_size(0);
    int np = pts.dim_size(1);
    int nf = faces.dim_size(1);
    int H = out_uvgrid.dim_size(1);
    int W = out_uvgrid.dim_size(2);

    Tensor *grad_pts_ptr = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{batch_size, np, 3},
                                            &grad_pts_ptr));
    auto grad_pts_data = grad_pts_ptr->flat<float>().data();
    rasterize_grad_impl(batch_size, nf, np, H, W, pts_data, faces_data,
                        uvs_data, out_uvgrid_data, out_z_data, out_fids_data,
                        out_bc_data, grad_uvgrid_data, grad_z_data,
                        grad_pts_data);
  }
};