#include "example.h"

REGISTER_OP("Scaled")
    .Attr("scale: float")
    .Input("data: float32")
    .Output("scaled: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("Scaled").Device(DEVICE_CPU), ScaledOp<CPUDevice>)
