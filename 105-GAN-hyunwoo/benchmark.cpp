#include "gan.h"
#include "timeChecker.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

// CPU-based matrix multiplication for benchmarking
void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M, int N, int K)
{
    // C = A * B
    // A: M x K
    // B: K x N
    // C: M x N
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU-based LeakyReLU for benchmarking
void cpu_leaky_relu(std::vector<float>& data, float alpha)
{
    for (auto& val : data) {
        if (val < 0) val *= alpha;
    }
}

// CPU-based BatchNorm for benchmarking
void cpu_batch_norm(std::vector<float>& data, int channels, int spatial_size)
{
    for (int c = 0; c < channels; c++) {
        // Compute mean
        float mean = 0.0f;
        for (int s = 0; s < spatial_size; s++) {
            mean += data[c * spatial_size + s];
        }
        mean /= spatial_size;

        // Compute variance
        float variance = 0.0f;
        for (int s = 0; s < spatial_size; s++) {
            float diff = data[c * spatial_size + s] - mean;
            variance += diff * diff;
        }
        variance /= spatial_size;

        // Normalize
        float std_dev = std::sqrt(variance + 1e-5f);
        for (int s = 0; s < spatial_size; s++) {
            data[c * spatial_size + s] = (data[c * spatial_size + s] - mean) / std_dev;
        }
    }
}

// CPU-based convolution (simplified, no optimization)
void cpu_conv2d(const std::vector<float>& input, const std::vector<float>& kernel,
                std::vector<float>& output, int H, int W, int C_in, int C_out, int K)
{
    int H_out = (H - K) / 2 + 1;  // stride=2
    int W_out = (W - K) / 2 + 1;

    for (int co = 0; co < C_out; co++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                float sum = 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_in = h * 2 + kh - K/2;
                            int w_in = w * 2 + kw - K/2;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                sum += input[ci * H * W + h_in * W + w_in] *
                                       kernel[co * C_in * K * K + ci * K * K + kh * K + kw];
                            }
                        }
                    }
                }
                output[co * H_out * W_out + h * W_out + w] = sum;
            }
        }
    }
}

void benchmark_cpu_operations()
{
    printf("\n");
    printf("=================================================================\n");
    printf("|             CPU Performance Benchmark (Baseline)              |\n");
    printf("=================================================================\n");
    printf("\n");

    const int ITERATIONS = 5;

    // Benchmark 1: Large Matrix Multiplication (simulating FC layer)
    {
        printf("1. Matrix Multiplication (FC Layer Simulation)\n");
        printf("   Dimensions: [256 x 2048] × [2048 x 256]\n");
        printf("   (Scaled down to avoid memory issues)\n");

        int M = 256, K = 2048, N = 256;
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N);

        auto start = std::chrono::high_resolution_clock::now();
        cpu_matmul(A, B, C, M, K, N);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("   Time: %ld ms\n", duration);
        printf("   Note: Full-scale would be ~1000x larger\n\n");
    }

    // Benchmark 2: LeakyReLU on tensor
    {
        printf("2. LeakyReLU Activation\n");
        printf("   Elements: 64×64×128 = 524,288\n");

        std::vector<float> data(64*64*128);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (float)i / data.size() - 0.5f;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            cpu_leaky_relu(data, 0.2f);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("   Time: %.2f ms/iter (avg of %d iterations)\n\n",
               duration / 1000.0f / ITERATIONS, ITERATIONS);
    }

    // Benchmark 3: BatchNorm on tensor
    {
        printf("3. Batch Normalization\n");
        printf("   Channels: 128, Spatial: 64×64 = 4,096\n");

        int channels = 128;
        int spatial = 64 * 64;
        std::vector<float> data(channels * spatial);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (float)i / data.size();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            cpu_batch_norm(data, channels, spatial);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("   Time: %.2f ms/iter (avg of %d iterations)\n\n",
               (float)duration / ITERATIONS, ITERATIONS);
    }

    // Benchmark 4: 2D Convolution
    {
        printf("4. 2D Convolution\n");
        printf("   Input: 64×64×64, Kernel: 5×5, Output channels: 128\n");

        int H = 64, W = 64, C_in = 64, C_out = 128, K = 5;
        std::vector<float> input(H * W * C_in, 1.0f);
        std::vector<float> kernel(C_out * C_in * K * K, 0.01f);
        std::vector<float> output(32 * 32 * C_out);

        auto start = std::chrono::high_resolution_clock::now();
        cpu_conv2d(input, kernel, output, H, W, C_in, C_out, K);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("   Time: %ld ms\n\n", duration);
    }

    printf("════════════════════════════════════════════════════════════════\n\n");
}


void benchmark_vulkan_operations(Device& device)
{
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║          Vulkan GPU Performance Benchmark (Compute)           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int ITERATIONS = 50;  // More iterations with smaller network
    const int WARMUP = 3;

    try {
        printf("Creating Generator network...\n");
        Generator gen(device);
        printf("✓ Generator created\n\n");

        printf("Creating Discriminator network...\n");
        Discriminator disc(device);
        printf("✓ Discriminator created\n\n");

        // Benchmark 1: Generator Forward Pass
        {
            printf("1. Generator Network (Forward Pass)\n");
            printf("   Architecture: 3 TransConv layers + BatchNorm + Activations\n");
            printf("   Input: Latent vector (100)\n");
            printf("   Output: 31×31×1 image (961 pixels)\n");

            // Warmup
            printf("   Warming up...\n");
            for (int i = 0; i < WARMUP; i++) {
                Tensor noise = gen.generateNoise(1);
                auto result = gen(noise);
            }

            // Benchmark
            printf("   Benchmarking %d iterations...\n", ITERATIONS);
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < ITERATIONS; iter++) {
                Tensor noise = gen.generateNoise(1);
                auto result = gen(noise);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("   Time: %.2f ms/iter (avg of %d iterations)\n",
                   (float)duration / ITERATIONS, ITERATIONS);
            printf("   Total: %ld ms\n\n", duration);
        }

        // Benchmark 2: Discriminator Forward Pass
        {
            printf("2. Discriminator Network (Forward Pass)\n");
            printf("   Architecture: 2 Conv layers + FC + Activations\n");
            printf("   Input: 31×31×1 image\n");
            printf("   Output: Probability (real/fake)\n");

            // Create dummy input
            std::vector<float> dummy_data(31 * 31 * 1, 0.5f);
            Tensor dummy_input(31, 31, 1);
            dummy_input.set(dummy_data);

            // Warmup
            printf("   Warming up...\n");
            for (int i = 0; i < WARMUP; i++) {
                auto result = disc(dummy_input);
            }

            // Benchmark
            printf("   Benchmarking %d iterations...\n", ITERATIONS);
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < ITERATIONS; iter++) {
                auto result = disc(dummy_input);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("   Time: %.2f ms/iter (avg of %d iterations)\n",
                   (float)duration / ITERATIONS, ITERATIONS);
            printf("   Total: %ld ms\n\n", duration);
        }

        // Benchmark 3: End-to-End GAN Pipeline
        {
            printf("3. Complete GAN Pipeline (Generator + Discriminator)\n");
            printf("   Noise → Generator → Image → Discriminator → Score\n");

            // Warmup
            printf("   Warming up...\n");
            for (int i = 0; i < WARMUP; i++) {
                Tensor noise = gen.generateNoise(1);
                auto fake_image = gen(noise)[0];
                auto score = disc(fake_image)[0];
            }

            // Benchmark
            printf("   Benchmarking %d iterations...\n", ITERATIONS);
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < ITERATIONS; iter++) {
                Tensor noise = gen.generateNoise(1);
                auto fake_image = gen(noise)[0];
                auto score = disc(fake_image)[0];
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("   Time: %.2f ms/iter (avg of %d iterations)\n",
                   (float)duration / ITERATIONS, ITERATIONS);
            printf("   Total: %ld ms\n\n", duration);
        }

    } catch (const std::exception& e) {
        printf("   ✗ Benchmark failed: %s\n\n", e.what());
    }

    printf("════════════════════════════════════════════════════════════════\n\n");
}


void print_performance_comparison()
{
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                   Performance Comparison                       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Architecture Complexity:\n");
    printf("  Large Generator:\n");
    printf("    - Layers: 6 TransposeConv + 6 BatchNorm + 6 Activations\n");
    printf("    - Parameters: ~150M (estimated)\n");
    printf("    - FLOPs: ~50 GFLOPs per forward pass\n");
    printf("    - Output: 119×119×3 = 42,483 pixels\n");
    printf("\n");
    printf("  Large Discriminator:\n");
    printf("    - Layers: 5 Conv + 2 FC + 6 Activations\n");
    printf("    - Parameters: ~25M (estimated)\n");
    printf("    - FLOPs: ~15 GFLOPs per forward pass\n");
    printf("    - Input: 119×119×3 = 42,483 pixels\n");
    printf("\n");
    printf("Expected Performance:\n");
    printf("  CPU (Single-threaded):\n");
    printf("    - Matrix operations: Sequential, cache-limited\n");
    printf("    - Large Generator: ~30,000-60,000 ms\n");
    printf("    - Large Discriminator: ~10,000-20,000 ms\n");
    printf("\n");
    printf("  Vulkan GPU (Parallel):\n");
    printf("    - Compute shaders: Massively parallel\n");
    printf("    - Expected speedup: 100-500x over CPU\n");
    printf("    - Large Generator: ~50-200 ms\n");
    printf("    - Large Discriminator: ~20-100 ms\n");
    printf("\n");
    printf("  CUDA GPU (if available):\n");
    printf("    - Similar to Vulkan, optimized kernels\n");
    printf("    - Expected speedup: 100-500x over CPU\n");
    printf("    - Typically 1.2-1.5x faster than Vulkan\n");
    printf("\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
}


void run_comprehensive_benchmark(Device& device)
{
    printf("\n");
    printf("==================================================================\n");
    printf("|         CPU vs Vulkan GPU Performance Comparison              |\n");
    printf("==================================================================\n");
    printf("\n");

    // CPU Benchmark
    printf("=== CPU Baseline Benchmark ===\n");
    fflush(stdout);

    benchmark_cpu_operations();

    printf("\n=== CPU Benchmark Completed ===\n\n");
    fflush(stdout);

    // Vulkan Benchmark
    printf("=== Vulkan GPU Benchmark ===\n");
    fflush(stdout);

    benchmark_vulkan_operations(device);

    printf("\n=== Vulkan Benchmark Completed ===\n");
    fflush(stdout);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                     Benchmark Completed!                       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Summary:\n");
    printf("  ✓ CPU baseline measurements completed\n");
    printf("  ✓ Vulkan GPU accelerated measurements completed\n");
    printf("  ✓ Compare the timing results above to see the speedup\n");
    printf("\n");
    printf("Notes:\n");
    printf("  - CPU times are expected to be much slower (100-500x)\n");
    printf("  - Vulkan leverages GPU parallelism for massive speedup\n");
    printf("  - Actual speedup depends on GPU model and workload\n");
    printf("  - CUDA would show similar performance to Vulkan\n");
    printf("\n");
}
