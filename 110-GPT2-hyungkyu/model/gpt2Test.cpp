#include "gpt2.h"
#include "../core/error.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace vk;

// Global device and descriptor pool (defined in embeddingNodeTest.cpp)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

void testGPT2SmallForward()
{
    std::cout << "\n========== Test: GPT-2 Small Forward Pass ===========" << std::endl;

    // Use a smaller config for testing
    GPT2Config config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 64,      // Smaller for faster testing
        .num_heads = 4,     // Smaller for faster testing
        .num_layers = 2,    // Only 2 layers for testing
        .dropout = 0.0f
    };

    std::cout << "Config:" << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Max seq len: " << config.max_seq_len << std::endl;
    std::cout << "  d_model: " << config.d_model << std::endl;
    std::cout << "  Num heads: " << config.num_heads << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;

    std::cout << "\nCreating GPT-2 model..." << std::endl;
    GPT2 model(netGlobalDevice, gDestSetPool, config);

    // Create dummy input: [batch=2, seq_len=4]
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 4;

    std::vector<int> input_ids_int = {
        15496, 11, 995, 0,      // Batch 0: "Hello, world!"
        40, 1101, 4673, 13      // Batch 1: "I'm learning."
    };

    // Convert to float for Tensor
    std::vector<float> input_ids(input_ids_int.begin(), input_ids_int.end());

    Tensor inputTensor = Tensor(batch_size, seq_len);
    inputTensor.set(input_ids);

    std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]" << std::endl;
    std::cout << "Input token IDs:" << std::endl;
    std::cout << "  Batch 0: ";
    for (int i = 0; i < seq_len; ++i) {
        std::cout << input_ids_int[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Batch 1: ";
    for (int i = seq_len; i < 2 * seq_len; ++i) {
        std::cout << input_ids_int[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\nRunning forward pass..." << std::endl;
    std::cout << "  (This may take a moment...)" << std::endl;

    try {
        Tensor output = model.forward(inputTensor);
        std::cout << "  Forward pass completed successfully!" << std::endl;

        std::cout << "Output shape: [" << output.shape()[0] << ", "
                  << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "Expected: [" << batch_size << ", " << seq_len << ", " << config.d_model << "]" << std::endl;

        // Copy output back to CPU for inspection
        Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * config.d_model * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, output.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    // Check basic sanity
    std::cout << "\nVerifying output..." << std::endl;

    bool has_nan = false;
    bool has_inf = false;
    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 0; i < batch_size * seq_len * config.d_model; ++i) {
        if (std::isnan(data[i])) has_nan = true;
        if (std::isinf(data[i])) has_inf = true;
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    std::cout << "  Has NaN: " << (has_nan ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Has Inf: " << (has_inf ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Value range: [" << std::fixed << std::setprecision(4)
              << min_val << ", " << max_val << "]" << std::endl;

    // Print sample outputs for first token
    std::cout << "\n  First token output (first 10 values):" << std::endl;
    std::cout << "    ";
    for (int i = 0; i < std::min(10, (int)config.d_model); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

        if (!has_nan && !has_inf) {
            std::cout << "\n✓ GPT-2 forward pass PASSED - basic sanity checks OK" << std::endl;
        } else {
            std::cout << "\n✗ GPT-2 forward pass FAILED - numerical issues detected" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "\n✗ Forward pass failed with exception: " << e.what() << std::endl;
    }
}

void gpt2Test()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 Model Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testGPT2SmallForward();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All GPT-2 tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    }
}
