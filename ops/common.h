#pragma once

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define XINLINE __device__ __host__
#define XGLOBAL __global__
#define XDEVICE __device__
#define XSHARED __shared__
#define XINVOKE_KERNEL(kernel_name, grid_dim, block_dim, shared_bytes)         \
  kernel_name<<<grid_dim, block_dim, shared_bytes>>>
#else
#define XINLINE
#define XGLOBAL
#define XDEVICE
#define XSHARED
#define XINVOKE_KERNEL(kernel_name, grid_dim, block_dim, shared_bytes)         \
  kernel_name
#endif

using namespace tensorflow;

#ifndef __CUDACC__
template <typename T> XINLINE T atomicAdd(T *addr, T v) { return *addr += v; }
#endif

// borrowed from MXNet
const int kMemUnitBits = 5;
const int kMaxThreadsPerBlock = 1024;

/*! \brief number of units that can do synchronized update, half warp size */
const int kMemUnit = 1 << kMemUnitBits;
/*! \brief mask that could be helpful sometime */
const int kMemUnitMask = kMemUnit - 1;
/*! \brief suggested thread number(logscale) for mapping kernel */
const int kBaseThreadBits = 8;
/*! \brief suggested thread number for mapping kernel */
const int kBaseThreadNum = 1 << kBaseThreadBits;
/*! \brief maximum value of grid */
const int kMaxGridNum = 65535;
/*! \brief maximum value of grid within each dimension */
const int kMaxGridDim = 65535;
/*! \brief suggested grid number for mapping kernel */
const int kBaseGridNum = 1024;

// using CPUDevice = Eigen::ThreadPoolDevice;
// using GPUDevice = Eigen::GpuDevice;
struct CPUDevice {};
struct GPUDevice {};

template <typename xpu> struct Kernel;

template <> struct Kernel<CPUDevice> {
  template <typename OP, typename... Args>
  inline static void Launch(OP op, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_cores = omp_get_thread_num();
    if (omp_cores <= 1) {
      // Zero means not to use OMP, but don't interfere with external OMP
      // behavior
      for (int i = 0; i < N; ++i) {
        op(i, args...);
      }
    } else {
#pragma omp parallel for num_threads(omp_cores)
      for (int i = 0; i < N; ++i) {
        op(i, args...);
      }
    }
#else
    for (int i = 0; i < N; ++i) {
      op(i, args...);
    }
#endif
  }
};

#ifdef __CUDACC__
template <typename OP, typename... Args>
__global__ void generic_kernel(OP op, int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    op(i, args...);
  }
}

template <> struct Kernel<GPUDevice> {
  template <typename OP, typename... Args>
  inline static void Launch(OP op, int N, Args... args) {
    int ngrid =
        std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    generic_kernel<OP, Args...><<<ngrid, kBaseThreadNum, 0>>>(op, N, args...);
  }
};
#endif // __CUDACC__
