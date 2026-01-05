#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int count, int res_x, int *output)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int threadx = blockDim.x * blockIdx.x + threadIdx.x;
    int thready = blockDim.y * blockIdx.y + threadIdx.y;
    float z_re = lower_x + (float)threadx * step_x;
    float z_im = lower_y + (float)thready * step_y;
    float c_re = z_re;
    float c_im = z_im;
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
    int idx = res_x * thready + threadx;
    output[idx] = i;
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

    dim3 threads(32, 32);
    dim3 blocks((res_x + threads.x - 1) / threads.x, (res_y + threads.y - 1) / threads.y);

    int *output;
    int *host_output = new int[res_x * res_y];
    
    cudaMalloc((void **)&output, n * sizeof(int));

    mandel_kernel<<<blocks, threads>>>(lower_x, lower_y, step_x, step_y, max_iterations, res_x, output);
    
    cudaMemcpy(host_output, output, n * sizeof(int), cudaMemcpyDeviceToHost);

    memcpy(img, host_output, res_x * res_y * sizeof(int));

    cudaFree(output);
}
