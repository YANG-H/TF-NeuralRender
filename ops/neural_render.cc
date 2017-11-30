#include "neural_render.h"

REGISTER_OP("Rasterize")
    .Attr("H: int")
    .Attr("W: int")
    .Input("pts: float32")
    .Input("faces: int32")
    .Input("uvs: float32")
    .Output("out_uvgrid: float32")
    .Output("out_fids: int32")
    .Output("out_z: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto batch_size = c->Dim(c->input(0), 0);
      int32_t H, W;
      GetNodeAttr(c->attrs(), "H", &H);
      GetNodeAttr(c->attrs(), "W", &W);
      c->set_output(0, c->MakeShape({batch_size, H, W, 2}));
      auto out_single_shape = c->MakeShape({batch_size, H, W});
      c->set_output(1, out_single_shape);
      c->set_output(2, out_single_shape);
      return Status::OK();
    });

// REGISTER_KERNEL_BUILDER(Name("Rasterize").Device(DEVICE_CPU),
//                        NeuralRenderOp<CPUDevice>)

// void rasterize_impl(CPUDevice, int batch_size, int npoints, int nfaces,
//                        const float *pts_data, const int32_t *faces_data,
//                        const float *uvs_data, int H, int W,
//                        float *out_uvgrid_data, int32_t *out_fids_data,
//                        float *out_z_data) {
//  // TODO
//}
