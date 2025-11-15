#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H

#include "neuralNet.h"

// Forward inference nodes (from 11-mnist-refactor)
class ConvolutionNode : public Node
{
    uint32_t C, F, K;   // C: input channels, F: output channels, K: kernel width

    ComputePipeline im2col;
    ComputePipeline gemm;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

public:
    ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class ReluNode : public Node
{
    ComputePipeline relu;
    DescriptorSet reluDescSet;

public:
    ReluNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: LeakyReLU for GAN (better gradient flow than ReLU)
class LeakyReluNode : public Node
{
    float alpha; // Slope for negative values (typically 0.2)
    ComputePipeline leakyRelu;
    DescriptorSet leakyReluDescSet;

public:
    LeakyReluNode(float alpha = 0.2f);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: Sigmoid activation for discriminator output
class SigmoidNode : public Node
{
    ComputePipeline sigmoid;
    DescriptorSet sigmoidDescSet;

public:
    SigmoidNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: Tanh activation for generator output (normalizes to [-1, 1])
class TanhNode : public Node
{
    ComputePipeline tanh;
    DescriptorSet tanhDescSet;

public:
    TanhNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: Batch Normalization (critical for GAN stability)
class BatchNormNode : public Node
{
    uint32_t numFeatures;
    float epsilon;
    ComputePipeline batchnorm;
    DescriptorSet batchnormDescSet;

public:
    BatchNormNode(uint32_t numFeatures, float epsilon = 1e-5f);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: Transposed Convolution (upsampling for generator)
class TransposeConvNode : public Node
{
    uint32_t C, F, K, S;   // C: in channels, F: out channels, K: kernel, S: stride

    ComputePipeline transConv;
    DescriptorSet transConvDescSet;

public:
    TransposeConvNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride = 2);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class FlattenNode : public Node
{
    ComputePipeline copy;
    DescriptorSet copyDescSet;

public:
    FlattenNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// New: Reshape node (useful for generator: latent vector -> 2D feature maps)
class ReshapeNode : public Node
{
    std::vector<uint32_t> targetShape;

public:
    ReshapeNode(const std::vector<uint32_t>& shape);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class FullyConnectedNode : public Node
{
    uint32_t I, O; // I: input size, O: output size
    ComputePipeline gemm;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

    ComputePipeline setZero;
    DescriptorSet setZeroDescSet;

public:
    FullyConnectedNode(uint32_t inDim, uint32_t outDim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


extern Device netGlobalDevice; // Global device for neural network operations


#endif // NEURAL_NODES_H
