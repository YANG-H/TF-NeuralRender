#pragma once

#include "common.h"

// void rasterize_impl(CPUDevice, int batch_size, int npoints, int nfaces,
//                        const float *pts_data, const int32_t *faces_data,
//                        const float *uvs_data, int H, int W,
//                        float *out_uvgrid_data, int32_t *out_fids_data,
//                        float *out_z_data);
void rasterize_impl(GPUDevice, int batch_size, int npoints, int nfaces,
                    const float *pts_data, const int32_t *faces_data,
                    const float *uvs_data, int H, int W, float *out_uvgrid_data,
                    int32_t *out_fids_data, float *out_z_data);

// out_uvgrid, out_fids, out_z = rasterize(pts, faces, uvs, screen_shape)
template <typename Device> class RasterizeOp : public OpKernel {
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

    Tensor *out_uvgrid_ptr = nullptr;
    Tensor *out_fids_ptr = nullptr;
    Tensor *out_z_ptr = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1), 2},
                       &out_uvgrid_ptr));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1)},
                       &out_fids_ptr));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       2, TensorShape{batch_size, screen_shape_.dim_size(0),
                                      screen_shape_.dim_size(1)},
                       &out_z_ptr));
    auto out_uvgrid_data = out_uvgrid_ptr->flat<float>().data();
    auto out_fids_data = out_fids_ptr->flat<int32_t>().data();
    auto out_z_data = out_z_ptr->flat<float>().data();

    int H = screen_shape_.dim_size(0);
    int W = screen_shape_.dim_size(1);
    rasterize_impl(Device(), batch_size, np, nf, pts_data, faces_data, uvs_data,
                   H, W, out_uvgrid_data, out_fids_data, out_z_data);
  }

private:
  TensorShape screen_shape_;
};

// grad(out_uvgrid, out_tex, out_fids, pts, faces, tex)
template <typename Device> class NeuralRenderGradOp : public OpKernel {
public:
  explicit NeuralRenderGradOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &pts = context->input(0);
    const Tensor &faces = context->input(1);
    const Tensor &uvs = context->input(2);
  }
};
