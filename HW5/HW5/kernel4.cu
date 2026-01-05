#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define STREAM_NUM 15
#define BLOCK_SIZE 8


__global__ __launch_bounds__(256, 2)
void mandel_kernel(float lower_x,
                   float lower_y,
                   float step_x,
                   float step_y,
                   int count,
                   int res_x,
                   int *output)
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
    if(count == 256){
        #pragma unroll
        for (i = 0; i < 256; ++i)
        {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;

            if (zr2 + zi2 > 4.f){
                goto EXIT;
            }

            float new_re = zr2 - zi2;
            float new_im = (z_re + z_re) * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }
    else if(count == 1000){
        #pragma unroll
        for (i = 0; i < 1000; ++i)
        {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;

            if (zr2 + zi2 > 4.f){
                goto EXIT;
            }

            float new_re = zr2 - zi2;
            float new_im = (z_re + z_re) * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }
    else if(count == 10000){
        #pragma unroll
        for (i = 0; i < 10000; ++i)
        {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;

            if (zr2 + zi2 > 4.f){
                goto EXIT;
            }

            float new_re = zr2 - zi2;
            float new_im = (z_re + z_re) * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }
    else if(count == 100000){
        #pragma unroll
        for (i = 0; i < 100000; ++i)
        {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;

            if (zr2 + zi2 > 4.f){
                goto EXIT;
            }

            float new_re = zr2 - zi2;
            float new_im = (z_re + z_re) * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }
    else{
        for (i = 0; i < count; ++i)
        {
            float zr2 = z_re * z_re;
            float zi2 = z_im * z_im;

            if (zr2 + zi2 > 4.f){
                goto EXIT;
            }

            float new_re = zr2 - zi2;
            float new_im = (z_re + z_re) * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }
EXIT:
    int idx = thready * res_x + threadx;
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

    dim3 threads(8, 8);
    dim3 blocks(
        (res_x + threads.x - 1) / threads.x,
        (res_y + threads.y - 1) / threads.y
    );

    size_t total_byte = n * sizeof(int);

    cudaHostRegister(img, total_byte, cudaHostRegisterDefault);

    int *output;
    cudaHostGetDevicePointer(&output, img, 0);

    mandel_kernel<<<blocks, threads>>>(lower_x, lower_y, step_x, step_y, max_iterations, res_x, output);

    cudaHostUnregister(img);
    cudaFree(output);
}
