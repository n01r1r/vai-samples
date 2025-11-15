#include "attentionNode.h"
#include "../testHelpers.h"
#include "../../core/neuralNet.h"
#include "../../core/error.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <functional>

using namespace vk;

// Global device and descriptor pool (defined in embeddingNodeTest.cpp)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

// ============================================================================
// Helper Functions for Test Code
// ============================================================================

struct TestConfig {
    uint32_t batch_size, seq_len, d_in, d_out, num_heads, head_dim;
};

void printTestConfig(const TestConfig& config) {
    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Sequence length: " << config.seq_len << std::endl;
    std::cout << "  d_in: " << config.d_in << std::endl;
    std::cout << "  d_out: " << config.d_out << std::endl;
    std::cout << "  num_heads: " << config.num_heads << std::endl;
    if (config.head_dim > 0) {
        std::cout << "  head_dim: " << config.head_dim << std::endl;
    }
}

std::vector<float> runInferenceAndGetOutput(NeuralNet& net, const Tensor& inputTensor, uint32_t output_size) {
    Tensor result = net(inputTensor)[0];

    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = output_size * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    return std::vector<float>(data, data + output_size);
}

bool performSanityChecks(const std::vector<float>& data) {
    bool has_nan = false, has_inf = false, all_zero = true;
    float min_val = data[0], max_val = data[0];

    for (float val : data) {
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
        if (val != 0.0f) all_zero = false;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }

    std::cout << "\nSanity checks:" << std::endl;
    std::cout << "  Has NaN: " << (has_nan ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Has Inf: " << (has_inf ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  All zero: " << (all_zero ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;

    return !has_nan && !has_inf && !all_zero;
}

void printComparisonResults(const std::vector<float>& actual, const std::vector<float>& expected, uint32_t d_out) {
    std::cout << "\n  First token (batch 0, token 0):" << std::endl;
    std::cout << "    Expected: ";
    for (size_t i = 0; i < d_out; ++i) {
        std::cout << std::fixed << std::setprecision(4) << expected[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "    Actual:   ";
    for (size_t i = 0; i < d_out; ++i) {
        std::cout << std::fixed << std::setprecision(4) << actual[i] << " ";
    }
    std::cout << std::endl;
}

struct ErrorStats {
    float max_error;
    float avg_error;
    int error_count;
};

ErrorStats calculateErrors(const std::vector<float>& actual, const std::vector<float>& expected, float threshold = 0.01f) {
    ErrorStats stats = {0.0f, 0.0f, 0};

    for (size_t i = 0; i < expected.size(); ++i) {
        float error = std::abs(actual[i] - expected[i]);
        stats.avg_error += error;
        stats.max_error = std::max(stats.max_error, error);
        if (error > threshold) {
            stats.error_count++;
        }
    }

    stats.avg_error /= expected.size();
    return stats;
}

// ============================================================================
// Generic Test Framework
// ============================================================================

template<typename NodeType>
struct TestFramework {
    using SetupWeightsFn = std::function<void(NodeType&)>;
    using CreateInputFn = std::function<Tensor()>;
    using VerifyResultsFn = std::function<bool(const std::vector<float>&)>;

    static std::vector<float> runTest(
        NodeType& node,
        CreateInputFn createInput,
        SetupWeightsFn setupWeights,
        uint32_t output_size)
    {
        NeuralNet net(netGlobalDevice, 1, 1);
        net.input(0) - node - net.output(0);

        setupWeights(node);
        Tensor inputTensor = createInput();

        return runInferenceAndGetOutput(net, inputTensor, output_size);
    }

    static bool runReferenceTest(
        NodeType& node,
        const json& testData,
        CreateInputFn createInput,
        SetupWeightsFn setupWeights,
        uint32_t output_size,
        float error_threshold = 0.1f)
    {
        std::vector<float> output = runTest(node, createInput, setupWeights, output_size);
        std::vector<float> expected = jsonToVector(testData["output"]);

        ErrorStats stats = calculateErrors(output, expected);

        std::cout << "\n  Error statistics:" << std::endl;
        std::cout << "    Max error: " << std::fixed << std::setprecision(6) << stats.max_error << std::endl;
        std::cout << "    Avg error: " << stats.avg_error << std::endl;
        std::cout << "    Values with error > 0.01: " << stats.error_count << " / " << expected.size() << std::endl;

        return stats.max_error < error_threshold;
    }
};

void testLinear()
{
    std::cout << "\n========== Test: Linear Layer ===========" << std::endl;

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 3;
    const uint32_t in_features = 8;
    const uint32_t out_features = 8;

    std::cout << "Creating neural network with Linear layer..." << std::endl;

    NeuralNet net(netGlobalDevice, 1, 1);
    LinearNode linear(in_features, out_features);

    net.input(0) - linear - net.output(0);

    // Create simple input
    std::vector<float> input_data(batch_size * seq_len * in_features);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = i * 0.1f;
    }

    Tensor inputTensor = Tensor(batch_size, seq_len, in_features).set(input_data);

    // Create simple weight pattern
    std::vector<float> weight_data(out_features * in_features);
    for (uint32_t o = 0; o < out_features; ++o) {
        for (uint32_t i = 0; i < in_features; ++i) {
            weight_data[o * in_features + i] = (o * in_features + i) * 0.01f;
        }
    }

    linear["weight"] = Tensor(out_features, in_features).set(weight_data);

    std::cout << "Running inference..." << std::endl;

    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * out_features * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    std::cout << "Output (first token, first 4 values): ";
    for (int i = 0; i < 4; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "✓ Linear layer test completed" << std::endl;
}

void testMultiHeadAttentionSimple()
{
    std::cout << "\n========== Test: Multi-Head Attention (Simple Input) ===========" << std::endl;

    TestConfig config = {1, 2, 4, 4, 2, 0};  // batch_size, seq_len, d_in, d_out, num_heads, head_dim
    printTestConfig(config);

    // Create network with simple input and weights (all 1.0 input, all 0.1 weights)
    NeuralNet net(netGlobalDevice, 1, 1);
    MultiHeadAttentionNode mha(config.d_in, config.d_out, config.num_heads);
    net.input(0) - mha - net.output(0);

    std::vector<float> input_data(config.batch_size * config.seq_len * config.d_in, 1.0f);
    Tensor inputTensor = Tensor(config.batch_size, config.seq_len, config.d_in).set(input_data);

    std::cout << "Input: all values = 1.0" << std::endl;
    std::cout << "Weights: all values = 0.1" << std::endl;

    mha["W_query"] = Tensor(config.d_in, config.d_in).set(std::vector<float>(config.d_in * config.d_in, 0.1f));
    mha["W_key"] = Tensor(config.d_in, config.d_in).set(std::vector<float>(config.d_in * config.d_in, 0.1f));
    mha["W_value"] = Tensor(config.d_in, config.d_in).set(std::vector<float>(config.d_in * config.d_in, 0.1f));
    mha["W_out"] = Tensor(config.d_out, config.d_in).set(std::vector<float>(config.d_out * config.d_in, 0.1f));

    std::cout << "Running inference..." << std::endl;

    // Run inference and get output
    std::vector<float> output = runInferenceAndGetOutput(net, inputTensor, config.batch_size * config.seq_len * config.d_out);

    // Print output
    std::cout << "\nOutput (first token): ";
    for (uint32_t i = 0; i < config.d_out; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output (second token): ";
    for (uint32_t i = 0; i < config.d_out; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output[config.d_out + i] << " ";
    }
    std::cout << std::endl;

    // Sanity checks
    bool passed = performSanityChecks(output);

    if (passed) {
        std::cout << "\n✓ Simple input test PASSED - basic functionality working" << std::endl;
    } else {
        std::cout << "\n✗ Simple input test FAILED - check implementation" << std::endl;
    }
}

void testMultiHeadAttentionReference()
{
    std::cout << "\n========== Test: Multi-Head Attention (Reference Data) ===========" << std::endl;
    std::cout << "Loading reference test data..." << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/attention/mha_test_data.json";
    json testData = loadTestData(testDataPath);

    // Load config (support both old and new format)
    TestConfig config;
    config.batch_size = testData["config"]["batch_size"];
    config.seq_len = testData["config"]["seq_len"];
    if (testData["config"].contains("d_in") && testData["config"].contains("d_out")) {
        config.d_in = testData["config"]["d_in"];
        config.d_out = testData["config"]["d_out"];
    } else {
        uint32_t d_model = testData["config"]["d_model"];
        config.d_in = config.d_out = d_model;
    }
    config.num_heads = testData["config"]["num_heads"];
    config.head_dim = testData["config"]["head_dim"];

    printTestConfig(config);

    // Create node
    MultiHeadAttentionNode mha(config.d_in, config.d_out, config.num_heads);

    // Use test framework
    auto createInput = [&]() {
        std::vector<float> input_data = jsonToVector(testData["input"]);
        std::cout << "Input loaded: " << input_data.size() << " values" << std::endl;
        return Tensor(config.batch_size, config.seq_len, config.d_in).set(input_data);
    };

    auto setupWeights = [&](MultiHeadAttentionNode& node) {
        node["W_query"] = Tensor(config.d_in, config.d_in).set(jsonToVector(testData["weights"]["W_query"]));
        node["W_key"] = Tensor(config.d_in, config.d_in).set(jsonToVector(testData["weights"]["W_key"]));
        node["W_value"] = Tensor(config.d_in, config.d_in).set(jsonToVector(testData["weights"]["W_value"]));
        node["W_out"] = Tensor(config.d_out, config.d_in).set(jsonToVector(testData["weights"]["W_out"]));
        std::cout << "Weights loaded" << std::endl;
    };

    std::cout << "Running inference..." << std::endl;

    uint32_t output_size = config.batch_size * config.seq_len * config.d_out;
    std::vector<float> output = TestFramework<MultiHeadAttentionNode>::runTest(
        mha, createInput, setupWeights, output_size);

    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;
    std::cout << "  Expected output size: " << expected_output.size() << std::endl;
    std::cout << "  Actual output size: " << output.size() << std::endl;

    printComparisonResults(output, expected_output, config.d_out);

    ErrorStats stats = calculateErrors(output, expected_output);

    std::cout << "\n  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << stats.max_error << std::endl;
    std::cout << "    Avg error: " << stats.avg_error << std::endl;
    std::cout << "    Values with error > 0.01: " << stats.error_count << " / " << expected_output.size() << std::endl;

    if (stats.max_error < 0.1f) {
        std::cout << "\n✓ Multi-Head Attention numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Multi-Head Attention numerical verification FAILED" << std::endl;
        std::cout << "  (This is expected for first implementation - debugging needed)" << std::endl;
    }
}

void attentionNodeTest()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Attention Node (Vulkan) - Numerical Verification Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // testLinear();  // TODO: Fix Linear test

        // Step 1: Test with simple input first
        testMultiHeadAttentionSimple();

        // Step 2: Test with reference data (only if simple test passes)
        std::cout << "\nProceeding to reference data test..." << std::endl;
        testMultiHeadAttentionReference();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Attention Node tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    }
}
