#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_SIZE 8

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandel_kernel(
        float lower_x,
        float lower_y,
        float step_x,
        float step_y,
        int count,
        int pitch,
        int group_size,
        int *output)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int threadx = blockIdx.x * blockDim.x + threadIdx.x;
    int thready = blockIdx.y * blockDim.y + threadIdx.y;

    int px = threadx * group_size;
    int py = thready * group_size;


    for (int j = 0; j < group_size; ++j){
        float y = lower_y + (j + py) * step_y;

        int pixel_y = thready * group_size + j;
        int *row = (int *)((char *)output + pixel_y * pitch);

        for (int k = 0; k < group_size; ++k){
            float x = lower_x + (k + px) * step_x;

            int pixel_x = threadx * group_size + k;

            row[pixel_x] = mandel(x, y, count);
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    // printf("%d %d\n", res_x, res_y);
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    const int n = res_x * res_y;

    int group_size = 2;
    dim3 threads(
        BLOCK_SIZE,
        BLOCK_SIZE
    );
    dim3 blocks(
        (res_x + group_size * BLOCK_SIZE - 1) / (group_size * BLOCK_SIZE),
        (res_y + group_size * BLOCK_SIZE - 1) / (group_size * BLOCK_SIZE)
    );

    int *output;
    int *host_output;
    size_t pitch = 0;
    size_t width_byte = res_x * sizeof(int);
    size_t total_byte = n * sizeof(int);

    cudaHostAlloc(&host_output, total_byte, cudaHostAllocDefault);
    
    cudaMallocPitch((void **)&output, &pitch, width_byte, res_y);

    mandel_kernel<<<blocks, threads>>>(lower_x, lower_y, step_x, step_y, max_iterations, pitch, group_size, output);
    
    cudaMemcpy2D(host_output, width_byte, output, pitch, width_byte, res_y, cudaMemcpyDeviceToHost);

    memcpy(img, host_output, total_byte);

    cudaFreeHost(host_output);
    cudaFree(output);
}
