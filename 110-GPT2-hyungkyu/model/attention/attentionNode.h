#ifndef ATTENTION_NODE_H
#define ATTENTION_NODE_H

#include "../../core/neuralNet.h"
#include "../../core/vulkanApp.h"

using namespace vk;

// Global device and descriptor pool (defined in test file)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

/**
 * Linear transformation: Y = X @ W^T
 * Input: [batch, seq_len, in_features]
 * Weight: [out_features, in_features]
 * Output: [batch, seq_len, out_features]
 */
class LinearNode : public Node
{
    uint32_t in_features;
    uint32_t out_features;

    ComputePipeline linearPipeline;
    DescriptorSet linearDescSet;

public:
    LinearNode(uint32_t in_features, uint32_t out_features);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Softmax along the last dimension
 * Input: [batch, ..., dim]
 * Output: [batch, ..., dim] (sum along last dim = 1.0)
 */
class SoftmaxNode : public Node
{
    ComputePipeline softmaxPipeline;
    DescriptorSet softmaxDescSet;

public:
    SoftmaxNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Multi-Head Attention
 * Input: [batch, seq_len, d_in]
 * Output: [batch, seq_len, d_out]
 *
 * Internal weights:
 * - W_query: [d_out, d_in] - project input to attention space
 * - W_key: [d_out, d_in] - project input to attention space
 * - W_value: [d_out, d_in] - project input to attention space
 * - W_out: [d_out, d_out] - final transformation in output space
 */
class MultiHeadAttentionNode : public Node
{
    uint32_t d_in;      // Input dimension
    uint32_t d_out;     // Output dimension
    uint32_t num_heads;
    uint32_t head_dim;

    // Pipelines for each stage
    ComputePipeline qkvProjection;        // Project input to Q, K, V
    ComputePipeline reshapeForHeads;      // Reshape to multi-head format
    ComputePipeline attentionScores;      // Q @ K^T / sqrt(head_dim)
    ComputePipeline applyCausalMask;      // Set upper triangle to -inf
    ComputePipeline softmaxPipeline;      // Softmax on attention scores
    ComputePipeline weightedSum;          // attn_weights @ V
    ComputePipeline combineHeads;         // Reshape and concat heads
    ComputePipeline outputProjection;     // Final linear projection

    // Descriptor sets
    DescriptorSet qkvProjDescSet;
    DescriptorSet reshapeDescSet;
    DescriptorSet scoresDescSet;
    DescriptorSet maskDescSet;
    DescriptorSet softmaxDescSet;
    DescriptorSet weightedSumDescSet;
    DescriptorSet combineDescSet;
    DescriptorSet outProjDescSet;

    // Helper struct for intermediate tensors
    struct IntermediateTensors {
        Tensor Q_flat, K_flat, V_flat;
        Tensor scores, attn_weights;
        Tensor context, context_combined;
    };

    // Private helper functions for better readability
    IntermediateTensors allocateIntermediateBuffers(uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD);
    void computeQKVProjection(CommandBuffer& cmdBuff, const Tensor& input, IntermediateTensors& tensors,
                              const Tensor& W_q, const Tensor& W_k, const Tensor& W_v, uint32_t B, uint32_t S, uint32_t D_in, uint32_t D_out);
    void computeAttentionScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S, uint32_t HD);
    void applyCausalMaskToScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S);
    void computeSoftmax(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S);
    void computeWeightedSum(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S, uint32_t HD);
    void combineHeadsAndProject(CommandBuffer& cmdBuff, IntermediateTensors& tensors, const Tensor& W_out, Tensor& output,
                                uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD);

public:
    MultiHeadAttentionNode(uint32_t d_in, uint32_t d_out, uint32_t num_heads);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

#endif // ATTENTION_NODE_H
