#pragma once

#ifdef _OPENMP
#include <omp.h>
int omp_get_thread_num();
#endif

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef GOOGLE_CUDA
#define XINLINE __device__ __host__
#else
#define XINLINE FORCE_INLINE
#endif

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

struct CPUDevice {};
struct GPUDevice {};

template <typename OP, typename xpu> struct Kernel;

template <typename OP> struct Kernel<OP, CPUDevice> {
  template <typename... Args>
  inline static void Launch(const int N, Args... args) {
#ifdef _OPENMP
    const int omp_cores = omp_get_thread_num();
    if (omp_cores <= 1) {
      // Zero means not to use OMP, but don't interfere with external OMP
      // behavior
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
#pragma omp parallel for num_threads(omp_cores)
      for (int i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (int i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
#endif
  }
};

#ifdef GOOGLE_CUDA
template <typename OP, typename... Args>
__global__ void generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template <typename OP> struct Kernel<OP, GPUDevice> {
  template <typename... Args> inline static void Launch(int N, Args... args) {
    int ngrid =
        std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    generic_kernel<OP, Args...><<<ngrid, kBaseThreadNum, 0>>>(N, args...);
  }
};
#endif // GOOGLE_CUDA
