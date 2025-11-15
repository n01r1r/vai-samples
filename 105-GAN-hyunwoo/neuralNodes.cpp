#include "neuralNodes.h"
#include <unordered_map>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Existing shaders from 11-mnist-refactor
static const char* src_relu = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = max(in0[o], 0.0f);
})";

// New: LeakyReLU shader for GAN
static const char* src_leakyRelu = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
    float alpha;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    float x = in0[o];
    out0[o] = x > 0.0f ? x : alpha * x;
})";

// New: Sigmoid shader
static const char* src_sigmoid = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = 1.0f / (1.0f + exp(-in0[o]));
})";

// New: Tanh shader
static const char* src_tanh = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    float ex = exp(2.0f * in0[o]);
    out0[o] = (ex - 1.0f) / (ex + 1.0f);
})";

// New: Batch Normalization shader (simplified for inference)
static const char* src_batchnorm = R"(
#version 450
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Gamma { float gamma[]; };
layout(set = 0, binding = 3) buffer Beta { float beta[]; };
layout(set = 0, binding = 4) buffer Mean { float mean[]; };
layout(set = 0, binding = 5) buffer Var { float var[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
    float epsilon;
};

void main()
{
    int h = int(gl_GlobalInvocationID.x);
    int w = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.z);

    if (h >= H || w >= W || c >= C) return;

    int idx = (h * W + w) * C + c;
    float normalized = (in0[idx] - mean[c]) / sqrt(var[c] + epsilon);
    out0[idx] = gamma[c] * normalized + beta[c];
})";

// New: Transpose Convolution (simplified - needs proper implementation)
static const char* src_transConv = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
layout(set = 0, binding = 3) buffer Bias { float bias[]; };
layout(push_constant) uniform PushConstants {
    int H_in;
    int W_in;
    int C_in;
    int H_out;
    int W_out;
    int C_out;
    int K;
    int S;
};

void main()
{
    int h_out = int(gl_GlobalInvocationID.x);
    int w_out = int(gl_GlobalInvocationID.y);
    int c_out = int(gl_GlobalInvocationID.z);

    if (h_out >= H_out || w_out >= W_out || c_out >= C_out) return;

    // Simplified transposed convolution
    // TODO: Implement proper transposed convolution
    float sum = bias[c_out];

    int K_2 = K / 2;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = (h_out + K_2 - kh) / S;
                int w_in = (w_out + K_2 - kw) / S;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    if ((h_out + K_2 - kh) % S == 0 && (w_out + K_2 - kw) % S == 0) {
                        int in_idx = (h_in * W_in + w_in) * C_in + c_in;
                        int w_idx = ((kh * K + kw) * C_in + c_in) * C_out + c_out;
                        sum += in0[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    out0[(h_out * W_out + w_out) * C_out + c_out] = sum;
})";

static const char* src_copy = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = in0[o];
})";

static const char* src_setZero = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = 0.0;
})";

static const char* src_im2col = R"(
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

    int h = i / W;
    int w = i % W;
    int c = j / KK;
    int K_2 = K / 2;
    int k = j % KK;

    float value = 0.0;
    h += k / K - K_2;
    w += k % K - K_2;
    if (0 <= h && h < H && 0 <= w && w < W)
        value = in0[((h * W) + w) * C + c];

    im2colOut[i * CKK + j] = value;
})";

static const char* src_gemm_shared = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

layout(push_constant) uniform PushConstants {
    int M;
    int K;
    int N;
};

shared float As[32 * 32];
shared float Bs[32 * 32];

void main()
{
    int n = int(gl_GlobalInvocationID.x);
    int m = int(gl_GlobalInvocationID.y);
    int _n = int(gl_LocalInvocationID.x);
    int _m = int(gl_LocalInvocationID.y);
    bool validThread = (m < M && n < N);

    float acc = 0.0;
    int sharedIdx = _m * 32 + _n;
    for (int k0 = 0; k0 < K; k0 += 32)
    {
        int n_ = k0 + _n;
        int m_ = k0 + _m;
        As[sharedIdx] = (m < M && n_ < K) ? A[m * K + n_] : 0.0;
        Bs[sharedIdx] = (m_ < K && n < N) ? B[m_ * N + n] : 0.0;
        barrier();

        for (int k = 0; k < 32; ++k)
            acc += As[_m * 32 + k] * Bs[k * 32 + _n];
        barrier();
    }

    if (validThread)
        C[m * N + n] = acc + b[n];
})";

static const char* src_gemm_kSplit = R"(
#version 450
#define P 16
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

layout(push_constant) uniform PushConstants {
    int M;
    int K;
    int N;
};

void main()
{
    int n = int(gl_GlobalInvocationID.x);
    int m = int(gl_GlobalInvocationID.y);
    int Pid = int(gl_GlobalInvocationID.z);

    if (n >= N) return;

    int k0 = Pid * P;
    float sum = (Pid==0) ? b[n] : 0.0;
    for (int p = 0; p < P; ++p)
    {
        int k = k0 + p;
        if (k >= K)
            break;
        sum += A[m * K + k] * B[k * N + n];
    }

    atomicAdd(C[m * N + n], sum);
}
)";

Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 200},
    .maxSets = 100
});

static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

static std::map<const char*, uint32_t> gGemmTileSize =
{
    {src_gemm_shared, 32},
};

void loadShaders()
{
    requestPipeline(src_relu);
    requestPipeline(src_leakyRelu);
    requestPipeline(src_sigmoid);
    requestPipeline(src_tanh);
    requestPipeline(src_batchnorm);
    requestPipeline(src_transConv);
    requestPipeline(src_copy);
    requestPipeline(src_setZero);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_shared);
    requestPipeline(src_gemm_kSplit);
}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth)
:  C(inChannels), F(outChannels), K(kernelWidth)
{
    _ASSERT(K % 2 == 1);
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("im2colOut", NodeSlot::internal);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    im2col = requestPipeline(src_im2col);
    im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);

    const char* gemmSrc = src_gemm_shared;

    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConvolutionNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    (*this)["im2colOut"] = Tensor(inShape[0], inShape[1], C*K*K);
    (*this)["out0"] = Tensor(inShape[0], inShape[1], F);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
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
    uint32_t M = H * W;
    uint32_t K_ = C * K * K;
    uint32_t N = F;
    uint32_t gemmConstants[] = {M, K_, N};

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
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// ReluNode
/////////////////////////////////////////////////////////////////////////////////////////
ReluNode::ReluNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    relu = requestPipeline(src_relu);
    reluDescSet = relu.descSetLayout(0).newDescSet(gDestSetPool);
}

void ReluNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void ReluNode::run(CommandBuffer cmdBuff)
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

/////////////////////////////////////////////////////////////////////////////////////////
// LeakyReluNode (NEW for GAN)
/////////////////////////////////////////////////////////////////////////////////////////
LeakyReluNode::LeakyReluNode(float alpha_)
: alpha(alpha_)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    leakyRelu = requestPipeline(src_leakyRelu);
    leakyReluDescSet = leakyRelu.descSetLayout(0).newDescSet(gDestSetPool);
}

void LeakyReluNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void LeakyReluNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;

    leakyReluDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    struct { int O; float alpha; } constants = {I, alpha};

    cmdBuff
        .bindPipeline(leakyRelu)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({leakyReluDescSet})
        .dispatch(I)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// SigmoidNode (NEW for GAN)
/////////////////////////////////////////////////////////////////////////////////////////
SigmoidNode::SigmoidNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    sigmoid = requestPipeline(src_sigmoid);
    sigmoidDescSet = sigmoid.descSetLayout(0).newDescSet(gDestSetPool);
}

void SigmoidNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void SigmoidNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;

    sigmoidDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int sigmoidConstants[] = {I};

    cmdBuff
        .bindPipeline(sigmoid)
        .setPushConstants(0, sizeof(sigmoidConstants), sigmoidConstants)
        .bindDescSets({sigmoidDescSet})
        .dispatch(I)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// TanhNode (NEW for GAN)
/////////////////////////////////////////////////////////////////////////////////////////
TanhNode::TanhNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    tanh = requestPipeline(src_tanh);
    tanhDescSet = tanh.descSetLayout(0).newDescSet(gDestSetPool);
}

void TanhNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void TanhNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;

    tanhDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int tanhConstants[] = {I};

    cmdBuff
        .bindPipeline(tanh)
        .setPushConstants(0, sizeof(tanhConstants), tanhConstants)
        .bindDescSets({tanhDescSet})
        .dispatch(I)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// BatchNormNode (NEW for GAN - simplified for inference)
/////////////////////////////////////////////////////////////////////////////////////////
BatchNormNode::BatchNormNode(uint32_t numFeatures_, float epsilon_)
: numFeatures(numFeatures_), epsilon(epsilon_)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    addSlot("mean", NodeSlot::input);
    addSlot("var", NodeSlot::input);

    batchnorm = requestPipeline(src_batchnorm);
    batchnormDescSet = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
}

void BatchNormNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    _ASSERT(inShape[2] == numFeatures);
    _ASSERT((*this)["gamma"].isShapeOf(numFeatures));
    _ASSERT((*this)["beta"].isShapeOf(numFeatures));
    _ASSERT((*this)["mean"].isShapeOf(numFeatures));
    _ASSERT((*this)["var"].isShapeOf(numFeatures));

    (*this)["out0"] = Tensor(inShape);
}

void BatchNormNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    batchnormDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["var"].buffer(),
    });

    struct { int H, W, C; float epsilon; } constants = {(int)H, (int)W, (int)C, epsilon};

    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({batchnormDescSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(H, W, C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// TransposeConvNode (NEW for GAN - stub implementation)
/////////////////////////////////////////////////////////////////////////////////////////
TransposeConvNode::TransposeConvNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride)
: C(inChannels), F(outChannels), K(kernelWidth), S(stride)
{
    _ASSERT(K % 2 == 1);
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    transConv = requestPipeline(src_transConv);
    transConvDescSet = transConv.descSetLayout(0).newDescSet(gDestSetPool);
}

void TransposeConvNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    _ASSERT(inShape[2] == C);
    _ASSERT((*this)["weight"].isShapeOf(K*K*C, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    uint32_t H_in = inShape[0], W_in = inShape[1];
    uint32_t H_out = (H_in - 1) * S + K;
    uint32_t W_out = (W_in - 1) * S + K;

    (*this)["out0"] = Tensor(H_out, W_out, F);
}

void TransposeConvNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    const auto& outShape = (*this)["out0"].shape();

    uint32_t H_in = inShape[0], W_in = inShape[1];
    uint32_t H_out = outShape[0], W_out = outShape[1];

    transConvDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    struct { int H_in, W_in, C_in, H_out, W_out, C_out, K, S; } constants =
        {(int)H_in, (int)W_in, (int)C, (int)H_out, (int)W_out, (int)F, (int)K, (int)S};

    cmdBuff
        .bindPipeline(transConv)
        .bindDescSets({transConvDescSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(H_out, W_out, F)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// FlattenNode
/////////////////////////////////////////////////////////////////////////////////////////
FlattenNode::FlattenNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void FlattenNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    Tensor& outTensor = (*this)["out0"] = (*this)["in0"];
    outTensor.reshape(outTensor.numElements());
}

void FlattenNode::run(CommandBuffer cmdBuff)
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// ReshapeNode (NEW for GAN)
/////////////////////////////////////////////////////////////////////////////////////////
ReshapeNode::ReshapeNode(const std::vector<uint32_t>& shape)
: targetShape(shape)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void ReshapeNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    _ASSERT(targetShape.size() == 3); // Only support 3D reshape for now
    Tensor& outTensor = (*this)["out0"] = (*this)["in0"];
    outTensor.reshape(targetShape[0], targetShape[1], targetShape[2]);
}

void ReshapeNode::run(CommandBuffer cmdBuff)
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// FullyConnectedNode
/////////////////////////////////////////////////////////////////////////////////////////
FullyConnectedNode::FullyConnectedNode(uint32_t inDim, uint32_t outDim)
: I(inDim), O(outDim)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    const char* gemmSrc = src_gemm_kSplit;

    gemm = requestPipeline(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

    setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);
}

void FullyConnectedNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    _ASSERT((*this)["weight"].isShapeOf(I, O));
    _ASSERT((*this)["bias"].isShapeOf(O));
    (*this)["out0"] = Tensor(O);
}

void FullyConnectedNode::run(CommandBuffer cmdBuff)
{
    uint32_t M = 1;
    uint32_t K = (*this)["in0"].shape()[0];
    uint32_t N = (*this)["out0"].shape()[0];

    setZeroDescSet.write({
        (*this)["out0"].buffer(),
    });
    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t setZeroConstants[] = { N };
    uint32_t gemmConstants[] = {M, K, N};

    cmdBuff
        .bindPipeline(setZero)
        .bindDescSets({setZeroDescSet})
        .setPushConstants(0, sizeof(setZeroConstants), setZeroConstants)
        .dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
        )

        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(N, 32), M, CEIL_DIV(K, 16))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}
