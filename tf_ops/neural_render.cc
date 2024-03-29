#include "neural_render.h"

REGISTER_OP("Rasterize")
    .Attr("H: int")
    .Attr("W: int")
    .Input("pts: float32")
    .Input("faces: int32")
    .Input("uvs: float32")
    .Output("out_uvgrid: float32")
    .Output("out_z: float32")
    .Output("out_fids: int32")
    .Output("out_bc: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto batch_size = c->Dim(c->input(0), 0);
      int32_t H, W;
      GetNodeAttr(c->attrs(), "H", &H);
      GetNodeAttr(c->attrs(), "W", &W);
      c->set_output(0, c->MakeShape({batch_size, H, W, 2})); // out_uvgrid
      auto out_single_shape = c->MakeShape({batch_size, H, W});
      c->set_output(1, out_single_shape);                    // out_z
      c->set_output(2, out_single_shape);                    // out_fids
      c->set_output(3, c->MakeShape({batch_size, H, W, 3})); // out_bc
      return Status::OK();
    });

REGISTER_OP("RasterizeGrad")
    .Input("pts: float32")
    .Input("faces: int32")
    .Input("uvs: float32")
    .Input("out_fids: int32")
    .Input("out_bc: float32")
    .Input("grad_uvgrid: float32")
    .Input("grad_z: float32")
    .Output("grad_pts: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("BilinearSample")
    .Input("tex: float32")
    .Input("uvgrid: float32")
    .Output("out: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto batch_size = c->Dim(c->input(0), 0);
      auto H = c->Dim(c->input(1), 1);
      auto W = c->Dim(c->input(1), 2);
      auto Dt = c->Dim(c->input(0), 3);
      c->set_output(0, c->MakeShape({batch_size, H, W, Dt})); // out
      return Status::OK();
    });

REGISTER_OP("BilinearSampleGrad")
    .Input("tex: float32")
    .Input("uvgrid: float32")
    .Input("grad_sampled: float32")
    .Output("grad_tex: float32")
    .Output("grad_uvgrid: float32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });
