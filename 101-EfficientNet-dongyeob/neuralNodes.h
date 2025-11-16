#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H

#include "neuralNet.h"
#include <memory>

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


class MaxPoolingNode : public Node
{
    const bool discardTail = true; // If true, discard the tail elements that don't fit into the pooling window
    uint32_t P;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize);
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


// Depthwise Cionvolution Node for EfficientNet
class DepthwiseConvNode : public Node
{
    uint32_t C, K;   // C: channels, K: kernel width

    ComputePipeline depthwiseConv;
    DescriptorSet descSet;

public:
    DepthwiseConvNode(uint32_t channels, uint32_t kernelWidth);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class AddNode : public Node
{
    ComputePipeline add;
    DescriptorSet descSet;

public:
    AddNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class GlobalAvgPoolNode : public Node
{
    ComputePipeline globalAvgPool;
    DescriptorSet descSet;

public:
    GlobalAvgPoolNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class SigmoidNode : public Node
{
    ComputePipeline sigmoid;
    DescriptorSet descSet;

public:
    SigmoidNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class BatchNormNode : public Node
{
    ComputePipeline batchnorm;
    DescriptorSet descSet;
    float eps;

public:
    BatchNormNode(float epsilon = 1e-5f);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// DWConv + BN + Swish in a single node
class DepthwiseConvBNSwishNode : public Node
{
    uint32_t C, K;   // C: channels, K: kernel width

    ComputePipeline fused;
    DescriptorSet descSet;

public:
    DepthwiseConvBNSwishNode(uint32_t channels, uint32_t kernelWidth);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class ConvBNSwishNode : public Node
{
    uint32_t C, F, K;   // C: input channels, F: output channels, K: kernel width

    ComputePipeline im2col;
    ComputePipeline gemm_bn_swish;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

public:
    ConvBNSwishNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class SEBlockNode : public Node
{
    uint32_t C, se_reduce;   // C: channels, se_reduce: squeeze reduction channels

    ComputePipeline se;
    DescriptorSet descSet;

public:
    SEBlockNode(uint32_t channels, uint32_t seReduce);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// Composite nodes
struct MBConvConfig
{
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t expand_ratio;
    uint32_t kernel_size;
    uint32_t stride;
    float se_ratio;
};

class MBConvBlockNode : public NodeGroup
{
    MBConvConfig config;
    
    // Member nodes (will be initialized based on config)
    std::unique_ptr<ConvBNSwishNode> expandConv;      // Only if expand_ratio > 1
    std::unique_ptr<DepthwiseConvBNSwishNode> depthwiseConv;
    std::unique_ptr<SEBlockNode> seBlock;             // Only if se_ratio > 0
    std::unique_ptr<ConvolutionNode> projectConv;
    std::unique_ptr<BatchNormNode> projectBN;
    std::unique_ptr<AddNode> addNode;                 // Only if stride == 1 and same channels

public:
    MBConvBlockNode(const MBConvConfig& cfg);
    Tensor& operator[](const std::string& name);
};


extern Device netGlobalDevice; // Global device for neural network operations


#endif // NEURAL_NODES_H

