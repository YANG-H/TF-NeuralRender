#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"

REGISTER_KERNEL_BUILDER(Name("Scaled").Device(DEVICE_GPU), ScaledOp<GPUDevice>)
#endif
