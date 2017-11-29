#define EIGEN_USE_GPU
#include "neural_render.h"

__global__ void neural_render_kernel(int batch_size, int np, int nf,
                                     int texH, int texW, int texD,
                                     const float *pts_data,
                                     const int32_t *faces_data,
                                     const float *uvs_data, const float *tex_data,
                                     int H, int W, float *out_tex_data,
                                     int32_t *out_fids_data, float *out_z_data){
    extern __shared__ float z_at_this_pixel[];

}

void detail::neural_render_impl(GPUDevice, int batch_size, int np, int nf,
                                int texH, int texW, int texD,
                                const float *pts_data,
                                const int32_t *faces_data,
                                const float *uvs_data, const float *tex_data,
                                int H, int W, float *out_tex_data,
                                int32_t *out_fids_data, float *out_z_data) {
//  neural_render_kernel<<<>>>();
}
