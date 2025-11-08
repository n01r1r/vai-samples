#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"


inline const char* im2col_srcCode = R"(
#version 450
layout(local_size_x = 64, local_size_y = 16) in;
layout(set = 0, binding = 0) writeonly buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
    int K;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x); 
    int j = int(gl_GlobalInvocationID.y); 
    int KK = K * K;
    int CKK = C * KK;
    if (i >= H * W || j >= CKK) 
        return;

    int h = i / W;          // image center row
    int w = i % W;          // image center col
    int c = j / KK;         // image channel
    int K_2 = K / 2;
    int k = j % KK;

    float value = 0.0;
    h += k / K - K_2;  
    w += k % K - K_2;   
    if (0 <= h && h < H && 0 <= w && w < W) 
        value = in0[((h * W) + w) * C + c];

    im2colOut[i * CKK + j] = value;
})";

inline const char* gemm_srcCode = R"(
#version 450
layout(local_size_x = 64, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
layout(set = 0, binding = 3) buffer Bias { float bias[]; };

// out0(NxO) = in0(NxI) * weight(IxO) + bias(O)
layout(push_constant) uniform PushConstants {
    int N;
    int I;
    int O;
};

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int o = int(gl_GlobalInvocationID.y); 
    if (n >= N || o >= O) 
        return;

    float sum = bias[o];
    for (int i = 0; i < I; ++i)
        sum += in0[n * I + i] * weight[i * O + o];

    out0[n * O + o] = sum;
})";

inline const char* relu_srcCode = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int I;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= I) 
        return;

    out0[i] = max(in0[i], 0.0f);
})";

inline const char* copy_srcCode = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int I;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= I) 
        return;

    out0[i] = in0[i];
})";

inline const char* maxpool_srcCode = R"(
#version 450
#define FLT_MIN -3.402823466e+38
#define DISCARD_TAIL
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;      // input height
    int W;      // input width
    int C;
    int P;      // pooling size
};

void main()
{
    int h_ = int(gl_GlobalInvocationID.x);  // output row
    int w_ = int(gl_GlobalInvocationID.y);  // output col
    int c = int(gl_GlobalInvocationID.z);   // channel
#ifdef DISCARD_TAIL
    int H_ = H / P;  
    int W_ = W / P;  
#else
    int H_ = (H + P - 1) / P;
    int W_ = (W + P - 1) / P;
#endif
    if (h_ >= H_ || w_ >= W_ || c >= C)
        return;

    int h0 = h_ * P;  
    int w0 = w_ * P;     
    float maxVal = FLT_MIN;
    for (int dh=0; dh < P; ++dh) 
    {
        int h = h0 + dh;  
        for (int dw=0; dw < P; ++dw) 
        {
            int w = w0 + dw;

        #ifndef DISCARD_TAIL
            if (h < H && w < W) 
        #endif
            {
                maxVal = max(maxVal, in0[(h * W + w) * C + c]);
            }
        }
    }
    out0[(h_ * W_ + w_) * C + c] = maxVal;
})";


inline static Device gDevice = VulkanApp::get().createDevice({.supportPresent = false});
inline static DescriptorPool gDestSetPool = gDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 200}, 
    .maxSets = 100
});

class ConvolutionNode : public Node
{
    uint32_t C, F, K;   // C: input channels, F: output channels, K: kernel width

    ComputePipeline im2col;
    ComputePipeline gemm;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;

public:
    ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth)
    :  C(inChannels), F(outChannels), K(kernelWidth)
    {
        _ASSERT(K % 2 == 1);
        slots.try_emplace("in0", NodeSlot::input, this);
        slots.try_emplace("out0", NodeSlot::output, this);       
        slots.try_emplace("im2colOut", NodeSlot::internal, this);       
        slots.try_emplace("weight", NodeSlot::input, this); 
        slots.try_emplace("bias", NodeSlot::input, this); 

        im2col = gDevice.createComputePipeline({im2col_srcCode});
        gemm = gDevice.createComputePipeline({gemm_srcCode});
        im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);
        gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
        _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
        _ASSERT((*this)["bias"].isShapeOf(F));

        const auto& inShape = (*this)["in0"].shape();
        (*this)["im2colOut"] = Tensor(inShape[0], inShape[1], C*K*K);
        (*this)["out0"] = Tensor(inShape[0], inShape[1], F);
    }
    
    void run(CommandBuffer cmdBuff) override
    {
        const auto& inShape = (*this)["in0"].shape();
        uint32_t H = inShape[0], W = inShape[1];

        im2colDescSet.write({
            (*this)["im2colOut"].buffer(),
            (*this)["in0"].buffer(),
        });

        gemmDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["im2colOut"].buffer(),
            (*this)["weight"].buffer(),
            (*this)["bias"].buffer(),
        });

        uint32_t im2colConstants[] = {H, W, C, K};
        uint32_t gemmConstants[] = {H * W, C * K * K, F};

        cmdBuff
            .bindPipeline(im2col)
            .bindDescSets({im2colDescSet})
            .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
            .dispatch(H * W, C * K * K)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["im2colOut"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            )

            .bindPipeline(gemm)
            .bindDescSets({gemmDescSet})
            .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
            .dispatch(H * W, F)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["out0"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }  
};


class ReluNode : public Node
{
    ComputePipeline relu;
    DescriptorSet reluDescSet;

public:
    ReluNode()
    {
        slots.try_emplace("in0", NodeSlot::input, this);
        slots.try_emplace("out0", NodeSlot::output, this);

        relu = gDevice.createComputePipeline({relu_srcCode});
        reluDescSet = relu.descSetLayout(0).newDescSet(gDestSetPool);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].validShape());
        (*this)["out0"] = Tensor((*this)["in0"].shape());
    }
    
    void run(CommandBuffer cmdBuff) override
    {
        const auto& inShape = (*this)["in0"].shape();
        int I = 1;
        for (int dim : inShape) I *= dim;
        
        reluDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["in0"].buffer(),
        });

        int reluConstants[] = {I};

        cmdBuff
            .bindPipeline(relu)
            .setPushConstants(0, sizeof(reluConstants), reluConstants)
            .bindDescSets({reluDescSet})
            .dispatch(I)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["out0"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }  
};


class MaxPoolingNode : public Node
{
    const bool discardTail = true; // If true, discard the tail elements that don't fit into the pooling window
    uint32_t P;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize)
    : P(poolSize)
    {
        slots.try_emplace("in0", NodeSlot::input, this);
        slots.try_emplace("out0", NodeSlot::output, this);

        maxpool = gDevice.createComputePipeline({maxpool_srcCode});
        maxpoolDescSet = maxpool.descSetLayout(0).newDescSet(gDestSetPool);
    }

    void prepare() override
    {
        const auto& inShape = (*this)["in0"].shape();
        _ASSERT(inShape.size() == 3);
        uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

        if (discardTail)
            (*this)["out0"] = Tensor(H / P, W / P, C);
        else    
            (*this)["out0"] = Tensor((H + P - 1) / P, (W + P - 1) / P, C);

    }

    void run(CommandBuffer cmdBuff) override
    {
        const auto& inShape = (*this)["in0"].shape();
        uint32_t H = inShape[0], W = inShape[1], C = inShape[2];
        uint32_t H_ = discardTail ? H / P : (H + P - 1) / P;
        uint32_t W_ = discardTail ? W / P : (W + P - 1) / P;

        maxpoolDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["in0"].buffer(),
        });

        uint32_t maxpoolConstants[] = {H, W, C, P};

        cmdBuff
            .bindPipeline(maxpool)
            .bindDescSets({maxpoolDescSet})
            .setPushConstants(0, sizeof(maxpoolConstants), maxpoolConstants)
            .dispatch(H_, W_, C)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["out0"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }  
};


class FlattenNode : public Node
{
    ComputePipeline copy;
    DescriptorSet copyDescSet;

public:
    FlattenNode()
    {
        slots.try_emplace("in0", NodeSlot::input, this);
        slots.try_emplace("out0", NodeSlot::output, this);

        copy = gDevice.createComputePipeline({copy_srcCode});
        copyDescSet = copy.descSetLayout(0).newDescSet(gDestSetPool);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].validShape());
        (*this)["out0"] = Tensor((*this)["in0"].numElements());
    }

    void run(CommandBuffer cmdBuff) override
    {
        // cmdBuff
        //     .barrier(
        //         (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
        //         / (*this)["in0"].buffer()
        //         / (PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_READ)
        //     )
        //     .copyBuffer((*this)["out0"].buffer(), (*this)["in0"].buffer())
        //     .barrier(
        //         (PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_WRITE)
        //         / (*this)["out0"].buffer()
        //         / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        //     );

        const auto& inShape = (*this)["in0"].shape();
        int I = 1;
        for (int dim : inShape) I *= dim;
        
        copyDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["in0"].buffer(),
        });

        int copyConstants[] = {I};

        cmdBuff
            .bindPipeline(copy)
            .setPushConstants(0, sizeof(copyConstants), copyConstants)
            .bindDescSets({copyDescSet})
            .dispatch(I)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["out0"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }  
};


class FullyConnectedNode : public Node
{
    uint32_t I, O; // I: input size, O: output size
    ComputePipeline gemm;
    DescriptorSet gemmDescSet;

public:
    FullyConnectedNode(uint32_t inDim, uint32_t outDim)
    : I(inDim), O(outDim) 
    {
        slots.try_emplace("in0", NodeSlot::input, this);
        slots.try_emplace("out0", NodeSlot::output, this);
        slots.try_emplace("weight", NodeSlot::input, this); 
        slots.try_emplace("bias", NodeSlot::input, this); 

        gemm = gDevice.createComputePipeline({gemm_srcCode});
        gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
    }

    void prepare() override
    {
        _ASSERT((*this)["in0"].isShapeOf(I));
        _ASSERT((*this)["weight"].isShapeOf(I, O));
        _ASSERT((*this)["bias"].isShapeOf(O));
        (*this)["out0"] = Tensor(O); 
    }
    
    void run(CommandBuffer cmdBuff) override
    {
        uint32_t I = (*this)["in0"].shape()[0];
        uint32_t O = (*this)["out0"].shape()[0];

        gemmDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["in0"].buffer(),
            (*this)["weight"].buffer(),
            (*this)["bias"].buffer(),
        });

        uint32_t gemmConstants[] = {1, I, O};

        cmdBuff
            .bindPipeline(gemm)
            .bindDescSets({gemmDescSet})
            .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
            .dispatch(1, O)
            .barrier( 
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / (*this)["out0"].buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );

    }  
};






#endif // NEURAL_NODES_H