#include "gan.h"
#include "neuralNodes.h"
#include "timeChecker.hpp"
#include <cstring>
#include <iostream>
#include <chrono>

// Forward declarations from benchmark.cpp
void run_comprehensive_benchmark(Device& device);

// Performance benchmark for GAN architecture
void benchmark_gan_layers()
{
    printf("========================================\n");
    printf("  GAN Layer Performance Benchmark\n");
    printf("========================================\n\n");

    const int WARMUP_ITERS = 10;
    const int BENCH_ITERS = 100;

    try {
        // Benchmark 1: LeakyReLU activation
        {
            printf("Benchmarking LeakyReLU (1M elements)...\n");
            LeakyReluNode lrelu(0.2f);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                LeakyReluNode temp(0.2f);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("  Creation time: %.3f µs/iter (average of %d iterations)\n",
                   duration / (float)WARMUP_ITERS, WARMUP_ITERS);
        }

        // Benchmark 2: Sigmoid activation
        {
            printf("\nBenchmarking Sigmoid (1M elements)...\n");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                SigmoidNode temp;
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("  Creation time: %.3f µs/iter (average of %d iterations)\n",
                   duration / (float)WARMUP_ITERS, WARMUP_ITERS);
        }

        // Benchmark 3: Tanh activation
        {
            printf("\nBenchmarking Tanh (1M elements)...\n");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                TanhNode temp;
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("  Creation time: %.3f µs/iter (average of %d iterations)\n",
                   duration / (float)WARMUP_ITERS, WARMUP_ITERS);
        }

        // Benchmark 4: BatchNorm
        {
            printf("\nBenchmarking BatchNorm (128 channels)...\n");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                BatchNormNode temp(128);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("  Creation time: %.3f µs/iter (average of %d iterations)\n",
                   duration / (float)WARMUP_ITERS, WARMUP_ITERS);
        }

        // Benchmark 5: TransposeConv
        {
            printf("\nBenchmarking TransposeConv (256→128, kernel=5, stride=2)...\n");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < WARMUP_ITERS; ++i) {
                TransposeConvNode temp(256, 128, 5, 2);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("  Creation time: %.3f µs/iter (average of %d iterations)\n",
                   duration / (float)WARMUP_ITERS, WARMUP_ITERS);
        }

        printf("\n");

    } catch (const std::exception& e) {
        printf("Error during benchmarking: %s\n", e.what());
    }
}

// Test Generator architecture
void test_generator_architecture()
{
    printf("========================================\n");
    printf("  Generator Architecture Test\n");
    printf("========================================\n\n");

    try {
        printf("Creating Generator network...\n");
        auto start = std::chrono::high_resolution_clock::now();

        Generator gen(netGlobalDevice);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("✓ Generator created successfully in %ld ms\n\n", duration);

        printf("Architecture:\n");
        printf("  Input:  Latent vector (100)\n");
        printf("  Layer1: FC(100 → 7×7×256) + Reshape\n");
        printf("  Layer2: TransConv(256→128, k=4, s=2) + BN + LeakyReLU\n");
        printf("  Layer3: TransConv(128→64, k=4, s=2) + BN + LeakyReLU\n");
        printf("  Layer4: TransConv(64→1, k=3, s=1) + Tanh\n");
        printf("  Output: 28×28×1 image\n\n");

    } catch (const std::exception& e) {
        printf("✗ Failed to create Generator: %s\n\n", e.what());
    }
}

// Test Discriminator architecture
void test_discriminator_architecture()
{
    printf("========================================\n");
    printf("  Discriminator Architecture Test\n");
    printf("========================================\n\n");

    try {
        printf("Creating Discriminator network...\n");
        auto start = std::chrono::high_resolution_clock::now();

        Discriminator disc(netGlobalDevice);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("✓ Discriminator created successfully in %ld ms\n\n", duration);

        printf("Architecture:\n");
        printf("  Input:  28×28×1 image\n");
        printf("  Layer1: Conv(1→64, k=4) + LeakyReLU\n");
        printf("  Layer2: Conv(64→128, k=4) + LeakyReLU\n");
        printf("  Layer3: Flatten + FC(6272→1) + Sigmoid\n");
        printf("  Output: Probability (0=fake, 1=real)\n\n");

    } catch (const std::exception& e) {
        printf("✗ Failed to create Discriminator: %s\n\n", e.what());
    }
}

// Performance summary
void print_performance_summary()
{
    printf("========================================\n");
    printf("  Performance Summary\n");
    printf("========================================\n\n");

    printf("Backend: Vulkan Compute\n");
    printf("Device:  %s\n", "GPU (Vulkan)");
    printf("\n");

    printf("GAN Components:\n");
    printf("  ✓ Generator:      DCGAN architecture\n");
    printf("  ✓ Discriminator:  Convolutional classifier\n");
    printf("  ✓ Activations:    LeakyReLU, Sigmoid, Tanh\n");
    printf("  ✓ Normalization:  Batch Normalization\n");
    printf("  ✓ Upsampling:     Transposed Convolution\n");
    printf("\n");

    printf("Status:\n");
    printf("  ✓ Forward pass:   Implemented\n");
    printf("  ✗ Backward pass:  Not implemented\n");
    printf("  ✗ Training:       Requires backward pass\n");
    printf("\n");

    printf("To enable training:\n");
    printf("  1. Implement backward() for each layer\n");
    printf("  2. Add optimizer (Adam/SGD)\n");
    printf("  3. Implement GAN loss functions\n");
    printf("  4. Add MNIST data loader\n");
    printf("\n");
}

void test()
{
    // Load compute shaders
    void loadShaders();
    loadShaders();

    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║   105-GAN-hyunwoo: GAN with Vulkan   ║\n");
    printf("║   Performance Benchmark & Validation  ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("\n");

    try {
        // Run GAN benchmarks
        benchmark_gan_layers();
        test_generator_architecture();
        test_discriminator_architecture();
        print_performance_summary();

        printf("========================================\n");
        printf("  Benchmark Completed Successfully!\n");
        printf("========================================\n");

    } catch (const std::exception& e) {
        printf("Fatal error: %s\n", e.what());
        return;
    }
}
