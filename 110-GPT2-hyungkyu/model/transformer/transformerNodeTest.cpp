#include "transformerNode.h"
#include "../attention/attentionNode.h"
#include "../testHelpers.h"
#include "../../core/neuralNet.h"
#include "../../core/error.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace vk;

// Global device and descriptor pool (defined in embeddingNodeTest.cpp)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

void testLayerNorm()
{
    std::cout << "\n========== Test: LayerNorm ===========" << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/transformer/layer_norm_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];
    float eps = testData["config"]["eps"];

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  eps: " << eps << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    LayerNormNode layerNorm(d_model, eps);

    net.input(0) - layerNorm - net.output(0);

    // Load input
    std::vector<float> input_data = jsonToVector(testData["input"]);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    // Load parameters
    std::vector<float> scale_data = jsonToVector(testData["scale"]);
    std::vector<float> shift_data = jsonToVector(testData["shift"]);

    layerNorm["scale"] = Tensor(d_model).set(scale_data);
    layerNorm["shift"] = Tensor(d_model).set(shift_data);

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
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

    // Load expected output
    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;

    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;

    for (size_t i = 0; i < expected_output.size(); ++i) {
        float error = std::abs(data[i] - expected_output[i]);
        avg_error += error;
        max_error = std::max(max_error, error);
    }

    avg_error /= expected_output.size();

    std::cout << "  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "    Avg error: " << avg_error << std::endl;

    if (max_error < 0.001f) {
        std::cout << "\n✓ LayerNorm numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ LayerNorm numerical verification FAILED" << std::endl;
    }
}

void testGELU()
{
    std::cout << "\n========== Test: GELU ===========" << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/transformer/gelu_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    GELUNode gelu;

    net.input(0) - gelu - net.output(0);

    // Load input
    std::vector<float> input_data = jsonToVector(testData["input"]);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
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

    // Load expected output
    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;

    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;

    for (size_t i = 0; i < expected_output.size(); ++i) {
        float error = std::abs(data[i] - expected_output[i]);
        avg_error += error;
        max_error = std::max(max_error, error);
    }

    avg_error /= expected_output.size();

    std::cout << "  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "    Avg error: " << avg_error << std::endl;

    if (max_error < 0.001f) {
        std::cout << "\n✓ GELU numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ GELU numerical verification FAILED" << std::endl;
    }
}

void testLinearSimple()
{
    std::cout << "\n========== Test: Linear (Simple) ===========" << std::endl;

    // Simple test to verify Linear layer works correctly
    const uint32_t batch_size = 1;
    const uint32_t seq_len = 2;
    const uint32_t in_features = 3;
    const uint32_t out_features = 4;

    // Create input: [[1, 2, 3], [4, 5, 6]]
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Create weight: identity-like for easy verification
    std::vector<float> weight_data(out_features * in_features);
    for (uint32_t o = 0; o < out_features; ++o) {
        for (uint32_t i = 0; i < in_features; ++i) {
            weight_data[o * in_features + i] = (i == o % in_features) ? 1.0f : 0.0f;
        }
    }

    NeuralNet net(netGlobalDevice, 1, 1);
    LinearNode linear(in_features, out_features);
    net.input(0) - linear - net.output(0);

    Tensor inputTensor = Tensor(batch_size, seq_len, in_features).set(input_data);
    linear["weight"] = Tensor(out_features, in_features).set(weight_data);

    Tensor result = net(inputTensor)[0];

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

    std::cout << "  Output:" << std::endl;
    for (uint32_t s = 0; s < seq_len; ++s) {
        std::cout << "    Token " << s << ": ";
        for (uint32_t o = 0; o < out_features; ++o) {
            std::cout << std::fixed << std::setprecision(2) << data[s * out_features + o] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "✓ Linear simple test completed" << std::endl;
}

void testLinear1FromFeedForward()
{
    std::cout << "\n========== Test: Linear1 (from FeedForward data) ===========" << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/transformer/feedforward_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];
    uint32_t hidden_dim = testData["config"]["hidden_dim"];

    std::cout << "Testing Linear1: [" << batch_size << "," << seq_len << "," << d_model << "] -> [" << batch_size << "," << seq_len << "," << hidden_dim << "]" << std::endl;

    // Load input and weight1
    std::vector<float> input_data = jsonToVector(testData["input"]);
    std::vector<float> weight1_data = jsonToVector(testData["weight1"]);
    std::vector<float> expected_hidden = jsonToVector(testData["intermediates"]["hidden"]);

    // Create Linear layer
    NeuralNet net(netGlobalDevice, 1, 1);
    LinearNode linear1(d_model, hidden_dim);
    net.input(0) - linear1 - net.output(0);

    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);
    linear1["weight"] = Tensor(hidden_dim, d_model).set(weight1_data);

    std::cout << "  Input shape: [" << inputTensor.shape()[0] << ", "
              << inputTensor.shape()[1] << ", " << inputTensor.shape()[2] << "]" << std::endl;

    Tensor result = net(inputTensor)[0];

    std::cout << "  Output shape: [" << result.shape()[0] << ", "
              << result.shape()[1] << ", " << result.shape()[2] << "]" << std::endl;
    std::cout << "  Output numElements: " << result.numElements() << std::endl;

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * hidden_dim * sizeof(float),
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

    std::cout << "  Expected total elements: " << expected_hidden.size() << std::endl;
    std::cout << "  Buffer size (floats): " << (batch_size * seq_len * hidden_dim) << std::endl;

    // Compare with expected
    float max_error = 0.0f;
    size_t max_error_idx = 0;
    size_t zero_count = 0;
    for (size_t i = 0; i < expected_hidden.size(); ++i) {
        float error = std::abs(data[i] - expected_hidden[i]);
        if (data[i] == 0.0f && expected_hidden[i] != 0.0f) zero_count++;
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }

    std::cout << "  Values incorrectly zero: " << zero_count << std::endl;

    // Print detailed pattern analysis
    std::cout << "  Pattern analysis (by batch-seq position):" << std::endl;
    for (uint32_t bs = 0; bs < batch_size * seq_len; ++bs) {
        size_t correct = 0, zero = 0;
        for (uint32_t o = 0; o < hidden_dim; ++o) {
            size_t idx = bs * hidden_dim + o;
            if (data[idx] == 0.0f && expected_hidden[idx] != 0.0f) {
                zero++;
            } else if (std::abs(data[idx] - expected_hidden[idx]) < 0.001f) {
                correct++;
            }
        }
        std::cout << "    bs=" << bs << " (batch=" << (bs / seq_len) << ", seq=" << (bs % seq_len)
                  << "): " << correct << " correct, " << zero << " incorrectly zero" << std::endl;
    }

    std::cout << "  First 8 hidden values:" << std::endl;
    std::cout << "    Expected: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::fixed << std::setprecision(6) << expected_hidden[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    Actual:   ";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Max error: " << max_error << " at index " << max_error_idx << std::endl;
    std::cout << "    Expected[" << max_error_idx << "]: " << expected_hidden[max_error_idx] << std::endl;
    std::cout << "    Actual[" << max_error_idx << "]:   " << data[max_error_idx] << std::endl;

    if (max_error < 0.001f) {
        std::cout << "✓ Linear1 test PASSED" << std::endl;
    } else {
        std::cout << "✗ Linear1 test FAILED" << std::endl;
    }
}

void testFeedForward()
{
    std::cout << "\n========== Test: FeedForward ===========" << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/transformer/feedforward_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];
    uint32_t hidden_dim = testData["config"]["hidden_dim"];

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  hidden_dim: " << hidden_dim << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    FeedForwardNode feedforward(d_model);

    net.input(0) - feedforward - net.output(0);

    // Load input
    std::vector<float> input_data = jsonToVector(testData["input"]);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    // Load weights
    std::vector<float> weight1_data = jsonToVector(testData["weight1"]);
    std::vector<float> weight2_data = jsonToVector(testData["weight2"]);

    feedforward["weight1"] = Tensor(hidden_dim, d_model).set(weight1_data);
    feedforward["weight2"] = Tensor(d_model, hidden_dim).set(weight2_data);

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
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

    // Load expected output
    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;

    // Print first few values for debugging
    std::cout << "  First token (batch 0, token 0):" << std::endl;
    std::cout << "    Expected: ";
    for (int i = 0; i < std::min(4, (int)d_model); ++i) {
        std::cout << std::fixed << std::setprecision(6) << expected_output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    Actual:   ";
    for (int i = 0; i < std::min(4, (int)d_model); ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
    }
    std::cout << std::endl;

    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int max_error_idx = 0;

    for (size_t i = 0; i < expected_output.size(); ++i) {
        float error = std::abs(data[i] - expected_output[i]);
        avg_error += error;
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }

    avg_error /= expected_output.size();

    std::cout << "\n  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "    Max error at index: " << max_error_idx << std::endl;
    std::cout << "      Expected: " << expected_output[max_error_idx] << std::endl;
    std::cout << "      Actual:   " << data[max_error_idx] << std::endl;
    std::cout << "    Avg error: " << avg_error << std::endl;

    if (max_error < 0.001f) {
        std::cout << "\n✓ FeedForward numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ FeedForward numerical verification FAILED" << std::endl;
    }
}

void testTransformerBlock()
{
    std::cout << "\n========== Test: TransformerBlock ===========" << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/transformer/transformer_block_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];
    uint32_t num_heads = testData["config"]["num_heads"];
    float eps = testData["config"]["eps"];

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  num_heads: " << num_heads << std::endl;
    std::cout << "  eps: " << eps << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    TransformerBlockNode transformerBlock(d_model, num_heads);

    net.input(0) - transformerBlock - net.output(0);

    // Load input
    std::vector<float> input_data = jsonToVector(testData["input"]);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    // Load weights
    auto weights = testData["weights"];
    transformerBlock["norm1_scale"] = Tensor(d_model).set(jsonToVector(weights["norm1_scale"]));
    transformerBlock["norm1_shift"] = Tensor(d_model).set(jsonToVector(weights["norm1_shift"]));
    transformerBlock["attn_wq"] = Tensor(d_model, d_model).set(jsonToVector(weights["W_query"]));
    transformerBlock["attn_wk"] = Tensor(d_model, d_model).set(jsonToVector(weights["W_key"]));
    transformerBlock["attn_wv"] = Tensor(d_model, d_model).set(jsonToVector(weights["W_value"]));
    transformerBlock["attn_wout"] = Tensor(d_model, d_model).set(jsonToVector(weights["W_out"]));
    transformerBlock["norm2_scale"] = Tensor(d_model).set(jsonToVector(weights["norm2_scale"]));
    transformerBlock["norm2_shift"] = Tensor(d_model).set(jsonToVector(weights["norm2_shift"]));
    transformerBlock["ff_w1"] = Tensor(4 * d_model, d_model).set(jsonToVector(weights["ff_weight1"]));
    transformerBlock["ff_w2"] = Tensor(d_model, 4 * d_model).set(jsonToVector(weights["ff_weight2"]));

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
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

    // Load expected output
    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;

    // Print first few values for debugging
    std::cout << "  First token (batch 0, token 0):" << std::endl;
    std::cout << "    Expected: ";
    for (int i = 0; i < std::min(4, (int)d_model); ++i) {
        std::cout << std::fixed << std::setprecision(6) << expected_output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    Actual:   ";
    for (int i = 0; i < std::min(4, (int)d_model); ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
    }
    std::cout << std::endl;

    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int max_error_idx = 0;

    for (size_t i = 0; i < expected_output.size(); ++i) {
        float error = std::abs(data[i] - expected_output[i]);
        avg_error += error;
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }

    avg_error /= expected_output.size();

    std::cout << "\n  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "    Max error at index: " << max_error_idx << std::endl;
    std::cout << "      Expected: " << expected_output[max_error_idx] << std::endl;
    std::cout << "      Actual:   " << data[max_error_idx] << std::endl;
    std::cout << "    Avg error: " << avg_error << std::endl;

    if (max_error < 0.001f) {
        std::cout << "\n✓ TransformerBlock numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ TransformerBlock numerical verification FAILED" << std::endl;
    }
}

void transformerNodeTest()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Transformer Node (Vulkan) - Numerical Verification Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // TODO: Fix LayerNorm crash
        // testLayerNorm();
        // testGELU();
        // testLinearSimple();
        testLinear1FromFeedForward();
        testFeedForward();
        testTransformerBlock();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Transformer Node tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    }
}
