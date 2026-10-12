#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define THREADS 256

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    __shared__ float shared[THREADS];

    // 1. Find maximum value in the row
    float local_max = -CUDART_INF_F;

    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(
                shared[tid],
                shared[tid + stride]
            );
        }
        __syncthreads();
    }

    float row_max = shared[0];

    // 2. Calculate sum(exp(x - max))
    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(x[i] - row_max);
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float denominator = shared[0];

    // 3. Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) / denominator;
    }
}