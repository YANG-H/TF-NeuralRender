#pragma once

#include "common.h"

template <typename Device> class ScaledOp : public OpKernel {
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
    const float *input_ptr = input_flat.data();
    auto output_flat = output.flat<float>();
    float *output_ptr = output_flat.data();

    float scale = scale_;
    Kernel<Device>::Launch([scale, input_ptr, output_ptr] XINLINE(
                               int i) { output_ptr[i] = input_ptr[i] * scale; },
                           input_flat.size());
  }

private:
  float scale_;
};