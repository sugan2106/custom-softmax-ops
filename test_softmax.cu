#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <random>

#include "softmax_kernel.cu"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

int main(int argc, char** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 1024;
    const int cols = argc > 2 ? atoi(argv[2]) : 1024;

    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Rows and columns must be positive.\n");
        return 1;
    }

    const size_t count = static_cast<size_t>(rows) * cols;
    const size_t bytes = count * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> output(count);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (float& value : input) {
        value = dist(rng);
    }

    std::ofstream input_file("input.bin", std::ios::binary);
    input_file.write(
        reinterpret_cast<const char*>(input.data()), bytes
    );
    input_file.close();

    float *d_input = nullptr, *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, input.data(), bytes, cudaMemcpyHostToDevice
    ));

    // Warm-up
    for (int i = 0; i < 20; ++i) {
        softmax_kernel<<<rows, THREADS>>>(
            d_input, d_output, rows, cols
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int REPEATS = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        softmax_kernel<<<rows, THREADS>>>(
            d_input, d_output, rows, cols
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(
        output.data(), d_output, bytes, cudaMemcpyDeviceToHost
    ));

    std::ofstream output_file("custom_output.bin", std::ios::binary);
    output_file.write(
        reinterpret_cast<const char*>(output.data()), bytes
    );
    output_file.close();

    std::ofstream timing_file("custom_latency.txt");
    timing_file << total_ms / REPEATS;
    timing_file.close();

    printf("GPU test shape: %d x %d\n", rows, cols);
    printf("Custom CUDA latency: %.6f ms\n", total_ms / REPEATS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}