#include "neural_render.h"

REGISTER_OP("NeuralRender")
    .Attr("screen_shape: TensorShape")
    .Input("pts: float32")
    .Input("faces: int32")
    .Input("uvs: float32")
    .Input("tex: float32")
    .Output("out_tex: float32")
    .Output("out_fids: int32")
    .Output("out_z: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size = c->Dim(c->input(0), 0);
      int texD = c->Dim(c->input(3), 4);
      TensorShape screen_shape;
      GetNodeAttr(c->attrs(), "screen_shape", &screen_shape);
      auto out_tex = c->MakeShape({batch_size, screen_shape.dim_size(0),
                                   screen_shape.dim_size(1), texD});
      c->set_output(0, out_tex);
      auto out = c->MakeShape(
          {batch_size, screen_shape.dim_size(0), screen_shape.dim_size(1)});
      c->set_output(1, out);
      c->set_output(2, out);
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("NeuralRender").Device(DEVICE_CPU),
                        NeuralRenderOp<CPUDevice>)

namespace detail {
void neural_render_impl(CPUDevice, int batch_size, int np, int nf, int texH,
                        int texW, int texD, const float *pts_data,
                        const int32_t *faces_data, const float *uvs_data,
                        const float *tex_data, int H, int W,
                        float *out_tex_data, int32_t *out_fids_data,
                        float *out_z_data) {
  // TODO
}
}
