#pragma once

#include <algorithm>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;

class ScaledOp : public OpKernel {
public:
  explicit ScaledOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    Tensor *optr = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &optr));
    Tensor &output = *optr;

    auto input_flat = input.flat<float>();
    const float *input_ptr = &(input_flat(0));
    auto output_flat = output.flat<float>();
    float *output_ptr = &(output_flat(0));

    std::transform(input_ptr, input_ptr + input_flat.size(), output_ptr,
                   [this](float v) { return v * scale_; });
  }

private:
  float scale_;
};