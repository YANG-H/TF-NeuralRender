#pragma once

#include "common.h"

namespace detail {
void neural_render_impl(GPUDevice, int batch_size, int np, int nf, int texH,
                        int texW, int texD, const float *pts_data,
                        const int32_t *faces_data, const float *uvs_data,
                        const float *tex_data, int H, int W,
                        float *out_tex_data, int32_t *out_fids_data,
                        float *out_z_data);
void neural_render_impl(CPUDevice, int batch_size, int np, int nf, int texH,
                        int texW, int texD, const float *pts_data,
                        const int32_t *faces_data, const float *uvs_data,
                        const float *tex_data, int H, int W,
                        float *out_tex_data, int32_t *out_fids_data,
                        float *out_z_data);
}

// rendering = neural_render(pts, faces, uvs, tex, screen_shape)
template <typename Device> class NeuralRenderOp : public OpKernel {
public:
  explicit NeuralRenderOp(OpKernelConstruction *context) : OpKernel(context) {
    context->GetAttr("screen_shape", &screen_shape_);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &pts = context->input(0); // BxNpx3, float, device coord
    auto pts_data = pts.flat<float>().data();
    const Tensor &faces = context->input(1); // BxNfx3, int
    auto faces_data = faces.flat<int32_t>().data();
    const Tensor &uvs = context->input(2); // BxNfx2, float
    auto uvs_data = uvs.flat<float>().data();
    const Tensor &tex = context->input(3); // BxNfxHxWxD, float
    auto tex_data = tex.flat<float>().data();

    int batch_size = pts.dim_size(0);
    //    OP_REQUIRES(context, batch_size == faces.dim_size(0),
    //                errors::InvalidArgument("batch sizes mismatch"));
    //    OP_REQUIRES(context, batch_size == uvs.dim_size(0),
    //                errors::InvalidArgument("batch sizes mismatch"));
    //    OP_REQUIRES(context, batch_size == tex.dim_size(0),
    //                errors::InvalidArgument("batch sizes mismatch"));

    int np = pts.dim_size(1);
    int nf = faces.dim_size(1);
    //    OP_REQUIRES(context, nf == tex.dim_size(1),
    //                errors::InvalidArgument("#faces mismatch"));
    int texH = tex.dim_size(2);
    int texW = tex.dim_size(3);
    int texD = tex.dim_size(4);

    Tensor *out_tex_ptr = nullptr;
    Tensor *out_fids_ptr = nullptr;
    Tensor *out_z_ptr = nullptr;
    OP_REQUIRES_OK(context->allocate_output(
        0, TensorShape{batch_size, screen_shape_.dim_size(0),
                       screen_shape_.dim_size(1), texD},
        &out_tex_ptr));
    OP_REQUIRES_OK(context->allocate_output(
        1, TensorShape{batch_size, screen_shape_.dim_size(0),
                       screen_shape_.dim_size(1)},
        &out_fids_ptr));
    OP_REQUIRES_OK(context->allocate_output(
        2, TensorShape{batch_size, screen_shape_.dim_size(0),
                       screen_shape_.dim_size(1)},
        &out_z_ptr));
    auto out_tex_data = out_tex_ptr->flat<float>().data();
    auto out_fids_data = out_fids_ptr->flat<int32_t>().data();
    auto out_z_data = out_z_ptr->flat<float>().data();

    int H = screen_shape_.dim_size(0);
    int W = screen_shape_.dim_size(1);
    //    int n = batch_size * nf * H * W;
    detail::neural_render_impl(Device(), batch_size, np, nf, texH, texW, texD,
                               pts_data, faces_data, uvs_data, tex_data, H, W,
                               out_tex_data, out_fids_data, out_z_data);

    //    // first compute visible faces at each screen pixel
    //    Kernel<Device>::Launch(
    //        [=] XINLINE(int i) {
    //          int x = i % W;
    //          int y = (i / W) % H;
    //          int fid = (i / W / H) % nf;
    //          int batch_id = (i / W / H / nf) % batch_size;

    //          // get 3 corner point positions
    //          int32_t pids[3];
    //          float ppos[3][3];
    //          for (int k = 0; k < 3; k++) {
    //            pids[k] = faces_data[(batch_id * nf + fid) * 3 + k];
    //            for (int j = 0; j < 3; j++) {
    //              ppos[k][j] = pts_data[(batch_id * np + pids[k]) * 3 + j];
    //            }
    //          }

    //          // get z depth here

    //        },
    //        n);
  }

private:
  TensorShape screen_shape_;
};

// grad
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
