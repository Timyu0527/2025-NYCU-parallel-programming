#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <xmmintrin.h>
#include "cycle_timer.h"

#define VECTOR_WIDTH 4

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);
static inline __m128 blendv_ps_sse2(__m128 a, __m128 b, __m128 mask){
    return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static inline __m128i vmandel(__m128 c_re, __m128 c_im, int count){
	__m128 z_re = c_re;
	__m128 z_im = c_im;
	const __m128 four = _mm_set1_ps(4.f);
	__m128i it = _mm_setzero_si128();

	for(int i = 0; i < count; ++i){
		__m128 z_re2 = _mm_mul_ps(z_re, z_re);
		__m128 z_im2 = _mm_mul_ps(z_im, z_im);
		__m128 mag2 = _mm_add_ps(z_re2, z_im2);

		__m128 alive = _mm_cmple_ps(mag2, four);
		int mask = _mm_movemask_ps(alive);
		if(!mask) break;

		__m128i inc = _mm_and_si128(_mm_set1_epi32(1), _mm_castps_si128(alive));
		it = _mm_add_epi32(it, inc);

		__m128 new_re = _mm_add_ps(_mm_sub_ps(z_re2, z_im2), c_re);
		__m128 prod   = _mm_mul_ps(z_re, z_im);
		__m128 new_im = _mm_add_ps(_mm_add_ps(prod, prod), c_im);

		z_re = blendv_ps_sse2(z_re, new_re, alive);
		z_im = blendv_ps_sse2(z_im, new_im, alive);
	}
	return it;
}

void mandelbrot_simd(
		float x0, float y0, float x1, float y1,
		int width, int height, 
		int start_row, int total_row,
		int max_iterations,
		int output[]){
	const float dx = (x1 - x0) / width;
	const float dy = (y1 - y0) / height;
	const int end_row = start_row + total_row;

	float *x_row = (float *)aligned_alloc(16, sizeof(float) * width);
	for(int i = 0; i < width; ++i) x_row[i] = x0 + i * dx;

	for(int j = start_row; j < end_row; ++j){
		const float y = y0 + j * dy;

		int i = 0;
		for(; i + VECTOR_WIDTH <= width; i += VECTOR_WIDTH){
			__m128 c_re = _mm_loadu_ps(x_row + i);
			__m128 c_im = _mm_set1_ps(y);
			__m128i it = vmandel(c_re, c_im, max_iterations);
			_mm_storeu_si128((__m128i*)&output[j*width + i], it);
		}


		int rem = width - i;
		if(rem > 0){
			float tmp_x[VECTOR_WIDTH] = {0};
			for (int k = 0; k < rem; ++k) tmp_x[k] = x_row[i + k];

			__m128 c_re = _mm_loadu_ps(tmp_x);
			__m128 c_im = _mm_set1_ps(y);
			__m128i it  = vmandel(c_re, c_im, max_iterations);

			int tmp_it[4];
			_mm_storeu_si128((__m128i*)tmp_it, it);
			for (int k = 0; k < rem; ++k){
				output[j*width + i + k] = tmp_it[k];
			}
		}

	}
	free(x_row);
}

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrot_serial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrot_serial() to this file and
    // modify it to pursue a better performance.
	int step = args->numThreads;
	int height = args->height;
	//int start_row = args->threadId * height / step;
	//int total_row = (step - 1== args->threadId) ? (height / step) + (height % step) : height / step;
	double startTime = CycleTimer::current_seconds();
	for(int i = args->threadId; i < height; i += step){
		mandelbrot_simd(args->x0, args->y0, args->x1, args->y1, args->width, args->height, i, 1, args->maxIterations, args->output);
	}
	//mandelbrot_simd(args->x0, args->y0, args->x1, args->y1, args->width, args->height, start_row, total_row, args->maxIterations, args->output);
	double endTime = CycleTimer::current_seconds();
	printf("[Thread %d]:\t\t[%.3f] ms\n", args->threadId, (endTime - startTime) * 1000);
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.


void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
