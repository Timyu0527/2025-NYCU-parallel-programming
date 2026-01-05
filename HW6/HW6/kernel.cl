__kernel void convolution(
    __global const float *input_image,   // 0
    __constant const float *filter,      // 1
    __global float *output_image,        // 2
    const int height,                    // 3
    const int width,                     // 4
    const int filter_width,              // 5
    __local float *tile                  // 6 (dynamic local memory)
){
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int localW = get_local_size(0);   // BLOCK
    int K = filter_width;
    int R = K / 2;

    int tileW = localW + K - 1;       // tile width including border

    int groupX = get_group_id(0) * localW;
    int groupY = get_group_id(1) * localW;

    // -----------------------------------------------------------
    // STEP 1: Load into local memory
    // -----------------------------------------------------------

    // We load tileW × tileW elements, more than threads available.
    for (int dy = ly; dy < tileW; dy += localW) {
        for (int dx = lx; dx < tileW; dx += localW) {

            //int ix4 = groupX + dx - R;
            int ix = groupX + dx - R;
            int iy = groupY + dy - R;

            //ix4 = clamp(ix4, 0, width - 4);
            ix = clamp(ix, 0, width - 1);
            iy = clamp(iy, 0, height - 1);

            tile[dy * tileW + dx] = input_image[iy * width + ix];
            /*
            float4 v = vload4(0, &input_image[iy * width + ix4]);

            int base = dy * tileW + dx;

            tile[base + 0] = v.s0;
            tile[base + 1] = v.s1;
            tile[base + 2] = v.s2;
            tile[base + 3] = v.s3;
            */
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // -----------------------------------------------------------
    // STEP 2: Convolution
    // -----------------------------------------------------------

    if (gx < width && gy < height) {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < K; ky++) {
            #pragma unroll
            for (int kx = 0; kx < K; kx++) {
                float v = tile[(ly + ky) * tileW + (lx + kx)];
                float w = filter[ky * K + kx];   // from constant memory
                sum += v * w;
            }
        }
        output_image[gy * width + gx] = sum;
    }
}

