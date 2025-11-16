#ifndef TRANSFORMER_NODE_H
#define TRANSFORMER_NODE_H

#include "../../core/neuralNet.h"
#include "../../core/vulkanApp.h"

using namespace vk;

// Global device and descriptor pool (defined in test file)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

/**
 * Layer Normalization
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Normalizes over the last dimension: mean=0, variance=1
 * Then applies learnable scale and shift parameters
 *
 * Formula: output = scale * (x - mean) / sqrt(var + eps) + shift
 */
class LayerNormNode : public Node
{
    uint32_t normalized_shape;  // d_model
    float eps;

    ComputePipeline layerNormPipeline;
    DescriptorSet layerNormDescSet;

public:
    LayerNormNode(uint32_t normalized_shape, float eps = 1e-5f);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * GELU Activation Function
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
class GELUNode : public Node
{
    ComputePipeline geluPipeline;
    DescriptorSet geluDescSet;

public:
    GELUNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Feed Forward Network (MLP)
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Architecture:
 * - Linear: d_model → 4*d_model
 * - GELU activation
 * - Linear: 4*d_model → d_model
 */
class FeedForwardNode : public Node
{
    uint32_t d_model;
    uint32_t hidden_dim;  // 4 * d_model

    ComputePipeline linear1Pipeline;
    ComputePipeline geluPipeline;
    ComputePipeline linear2Pipeline;

    DescriptorSet linear1DescSet;
    DescriptorSet geluDescSet;
    DescriptorSet linear2DescSet;

public:
    FeedForwardNode(uint32_t d_model);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Transformer Block
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Architecture (pre-norm):
 * - x = x + MultiHeadAttention(LayerNorm(x))
 * - x = x + FeedForward(LayerNorm(x))
 */
class TransformerBlockNode : public Node
{
    uint32_t d_model;
    uint32_t num_heads;

    // Pipelines for LayerNorm and ResidualAdd
    ComputePipeline norm1Pipeline;
    ComputePipeline residualAdd1Pipeline;
    ComputePipeline norm2Pipeline;
    ComputePipeline residualAdd2Pipeline;

    DescriptorSet norm1DescSet;
    DescriptorSet residualAdd1DescSet;
    DescriptorSet norm2DescSet;
    DescriptorSet residualAdd2DescSet;

    // We'll use the same shaders as LayerNormNode, FeedForwardNode, and MultiHeadAttentionNode
    // by including their implementations inline

public:
    TransformerBlockNode(uint32_t d_model, uint32_t num_heads);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

#endif // TRANSFORMER_NODE_H
