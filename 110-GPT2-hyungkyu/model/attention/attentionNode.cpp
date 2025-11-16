#include "attentionNode.h"
#include "../../core/error.h"
#include <cmath>
#include <unordered_map>

using namespace vk;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

// ============================================================================
// LinearNode: Y = X @ W^T
// ============================================================================

static const char* src_linear = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // [B*S*O]
layout(set = 0, binding = 1) buffer Input { float x[]; };       // [B*S*I]
layout(set = 0, binding = 2) buffer Weight { float w[]; };      // [O*I]

layout(push_constant) uniform PushConstants {
    int B;   // batch size
    int S;   // sequence length
    int I;   // in_features
    int O;   // out_features
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);  // batch * seq index
    int o = int(gl_GlobalInvocationID.y);   // output feature index

    int BS = B * S;
    if (bs >= BS || o >= O) return;

    float sum = 0.0;
    for (int i = 0; i < I; ++i) {
        sum += x[bs * I + i] * w[o * I + i];
    }

    y[bs * O + o] = sum;
}
)";

LinearNode::LinearNode(uint32_t in_features, uint32_t out_features)
    : in_features(in_features), out_features(out_features)
{
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::internal);
    addSlot("out0", NodeSlot::output);

    linearPipeline = requestPipeline(src_linear);
    linearDescSet = linearPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void LinearNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, I]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t I = input.shape()[2];
    _ASSERT(I == in_features);

    Tensor& weight = (*this)["weight"];
    if (!weight.validShape()) {
        weight = Tensor(out_features, in_features);
    }

    (*this)["out0"] = Tensor(B, S, out_features);
}

void LinearNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& weight = (*this)["weight"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t BS = B * S;

    linearDescSet.write({
        output.buffer(),
        input.buffer(),
        weight.buffer()
    });

    int constants[] = {(int)B, (int)S, (int)in_features, (int)out_features};

    cmdBuff
        .bindPipeline(linearPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({linearDescSet})
        .dispatch0(CEIL_DIV(BS, 16), CEIL_DIV(out_features, 16))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// SoftmaxNode: Numerically stable softmax
// ============================================================================

static const char* src_softmax = R"(
#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };
layout(set = 0, binding = 1) buffer Input { float x[]; };

layout(push_constant) uniform PushConstants {
    int num_rows;
    int row_size;
};

void main() {
    int row = int(gl_GlobalInvocationID.x);
    if (row >= num_rows) return;

    int offset = row * row_size;

    // Find max value
    float max_val = x[offset];
    for (int i = 1; i < row_size; ++i) {
        max_val = max(max_val, x[offset + i]);
    }

    // Compute exp and sum
    float sum_exp = 0.0;
    for (int i = 0; i < row_size; ++i) {
        sum_exp += exp(x[offset + i] - max_val);
    }

    // Normalize
    for (int i = 0; i < row_size; ++i) {
        y[offset + i] = exp(x[offset + i] - max_val) / sum_exp;
    }
}
)";

SoftmaxNode::SoftmaxNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    softmaxPipeline = requestPipeline(src_softmax);
    softmaxDescSet = softmaxPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void SoftmaxNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());

    // Output has same shape as input
    (*this)["out0"] = Tensor(input.shape());
}

void SoftmaxNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];

    auto shape = input.shape();
    int num_rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        num_rows *= shape[i];
    }
    int row_size = shape.back();

    softmaxDescSet.write({
        output.buffer(),
        input.buffer()
    });

    int constants[] = {num_rows, row_size};

    cmdBuff
        .bindPipeline(softmaxPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(num_rows, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// MultiHeadAttentionNode
// ============================================================================

// Shader 1: Project input to Q, K, V (3 separate linear projections)
static const char* src_qkv_projection = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Q { float q[]; };        // [B*S*D_out]
layout(set = 0, binding = 1) buffer K { float k[]; };        // [B*S*D_out]
layout(set = 0, binding = 2) buffer V { float v[]; };        // [B*S*D_out]
layout(set = 0, binding = 3) buffer Input { float x[]; };    // [B*S*D_in]
layout(set = 0, binding = 4) buffer Wq { float wq[]; };      // [D_out*D_in]
layout(set = 0, binding = 5) buffer Wk { float wk[]; };      // [D_out*D_in]
layout(set = 0, binding = 6) buffer Wv { float wv[]; };      // [D_out*D_in]

layout(push_constant) uniform PushConstants {
    int B, S, D_in, D_out;
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int d_out = int(gl_GlobalInvocationID.y);

    int BS = B * S;
    if (bs >= BS || d_out >= D_out) return;

    // Q = X @ Wq^T
    // X: [bs, D_in], Wq: [D_out, D_in] -> Q: [bs, D_out]
    float q_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        q_val += x[bs * D_in + i] * wq[d_out * D_in + i];
    }
    q[bs * D_out + d_out] = q_val;

    // K = X @ Wk^T
    float k_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        k_val += x[bs * D_in + i] * wk[d_out * D_in + i];
    }
    k[bs * D_out + d_out] = k_val;

    // V = X @ Wv^T
    float v_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        v_val += x[bs * D_in + i] * wv[d_out * D_in + i];
    }
    v[bs * D_out + d_out] = v_val;
}
)";

// Shader 2: Compute attention scores: Q @ K^T / sqrt(head_dim)
// Input Q, K: [B, S, d_in] where d_in = H * HD
// Output scores: [B, H, S, S]
static const char* src_attention_scores = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };  // [B*H*S*S]
layout(set = 0, binding = 1) buffer Q { float q[]; };             // [B*S*d_in]
layout(set = 0, binding = 2) buffer K { float k[]; };             // [B*S*d_in]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
    float scale;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);  // batch * head
    int s1 = int(gl_GlobalInvocationID.y);  // query position

    int BH = B * H;
    if (bh >= BH || s1 >= S) return;

    int b = bh / H;
    int h = bh % H;
    int D = H * HD;

    // Compute scores for all key positions
    for (int s2 = 0; s2 < S; ++s2) {
        float score = 0.0;
        for (int hd = 0; hd < HD; ++hd) {
            // Q[b, s1, h*HD + hd]
            float q_val = q[b * S * D + s1 * D + h * HD + hd];
            // K[b, s2, h*HD + hd]
            float k_val = k[b * S * D + s2 * D + h * HD + hd];
            score += q_val * k_val;
        }
        scores[bh * S * S + s1 * S + s2] = score * scale;
    }
}
)";

// Shader 3: Apply causal mask (set upper triangle to -inf)
static const char* src_causal_mask = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };

layout(push_constant) uniform PushConstants {
    int B, H, S;
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int BHS2 = B * H * S * S;

    if (idx >= BHS2) return;

    // Decode indices
    int bhs2 = idx;
    int s2 = bhs2 % S;
    bhs2 /= S;
    int s1 = bhs2 % S;

    // Apply causal mask: if s2 > s1, set to -inf
    if (s2 > s1) {
        scores[idx] = -1e38;  // -inf approximation
    }
}
)";

// Shader 4: Weighted sum: context = attn_weights @ V
// Input V: [B, S, d_in] where d_in = H * HD
// Output context: [B, H, S, HD]
static const char* src_weighted_sum = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Context { float ctx[]; };         // [B*H*S*HD]
layout(set = 0, binding = 1) buffer AttnWeights { float attn[]; };    // [B*H*S*S]
layout(set = 0, binding = 2) buffer V { float v[]; };                 // [B*S*d_in]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);
    int s = int(gl_GlobalInvocationID.y);

    int BH = B * H;
    if (bh >= BH || s >= S) return;

    int b = bh / H;
    int h = bh % H;
    int D = H * HD;

    // context[bh, s, :] = attn_weights[bh, s, :] @ V[bh, :, :]
    for (int hd = 0; hd < HD; ++hd) {
        float sum = 0.0;
        for (int s2 = 0; s2 < S; ++s2) {
            float weight = attn[bh * S * S + s * S + s2];
            // V[b, s2, h*HD + hd]
            float v_val = v[b * S * D + s2 * D + h * HD + hd];
            sum += weight * v_val;
        }
        ctx[bh * S * HD + s * HD + hd] = sum;
    }
}
)";

// Shader 5: Combine heads and reshape
// Input: [B, H, S, HD]
// Output: [B, S, d_in] where d_in = H * HD
static const char* src_combine_heads = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float out0[]; };     // [B*S*d_in]
layout(set = 0, binding = 1) buffer Context { float ctx[]; };     // [B*H*S*HD]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int d = int(gl_GlobalInvocationID.y);

    int BS = B * S;
    int D = H * HD;

    if (bs >= BS || d >= D) return;

    int b = bs / S;
    int s = bs % S;
    int h = d / HD;
    int hd = d % HD;

    // out[b, s, d] = context[b, h, s, hd]
    out0[bs * D + d] = ctx[b * H * S * HD + h * S * HD + s * HD + hd];
}
)";

MultiHeadAttentionNode::MultiHeadAttentionNode(uint32_t d_in, uint32_t d_out, uint32_t num_heads)
    : d_in(d_in), d_out(d_out), num_heads(num_heads)
{
    _ASSERT(d_out % num_heads == 0);
    head_dim = d_out / num_heads;

    addSlot("in0", NodeSlot::input);
    addSlot("W_query", NodeSlot::internal);
    addSlot("W_key", NodeSlot::internal);
    addSlot("W_value", NodeSlot::internal);
    addSlot("W_out", NodeSlot::internal);
    addSlot("out0", NodeSlot::output);

    // Create pipelines
    qkvProjection = requestPipeline(src_qkv_projection);
    attentionScores = requestPipeline(src_attention_scores);
    applyCausalMask = requestPipeline(src_causal_mask);
    softmaxPipeline = requestPipeline(src_softmax);
    weightedSum = requestPipeline(src_weighted_sum);
    combineHeads = requestPipeline(src_combine_heads);

    // Create descriptor sets
    qkvProjDescSet = qkvProjection.descSetLayout(0).newDescSet(gDestSetPool);
    scoresDescSet = attentionScores.descSetLayout(0).newDescSet(gDestSetPool);
    maskDescSet = applyCausalMask.descSetLayout(0).newDescSet(gDestSetPool);
    softmaxDescSet = softmaxPipeline.descSetLayout(0).newDescSet(gDestSetPool);
    weightedSumDescSet = weightedSum.descSetLayout(0).newDescSet(gDestSetPool);
    combineDescSet = combineHeads.descSetLayout(0).newDescSet(gDestSetPool);
}

void MultiHeadAttentionNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];  // Should be d_in
    _ASSERT(D == d_in);

    // Initialize weights if not set
    Tensor& W_q = (*this)["W_query"];
    Tensor& W_k = (*this)["W_key"];
    Tensor& W_v = (*this)["W_value"];
    Tensor& W_out = (*this)["W_out"];

    // Q, K, V projections: (d_out, d_in) - project input to attention space
    if (!W_q.validShape()) W_q = Tensor(d_out, d_in);
    if (!W_k.validShape()) W_k = Tensor(d_out, d_in);
    if (!W_v.validShape()) W_v = Tensor(d_out, d_in);
    // Output projection: (d_out, d_out) - final transformation in output space
    if (!W_out.validShape()) W_out = Tensor(d_out, d_out);

    (*this)["out0"] = Tensor(B, S, d_out);
}

void MultiHeadAttentionNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& W_q = (*this)["W_query"];
    Tensor& W_k = (*this)["W_key"];
    Tensor& W_v = (*this)["W_value"];
    Tensor& W_out = (*this)["W_out"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D_in = d_in;   // Input dimension
    uint32_t D_out = d_out; // Output/attention space dimension
    uint32_t H = num_heads;
    uint32_t HD = head_dim;

    // Allocate temporary buffers (Q, K, V are in d_out space)
    IntermediateTensors tensors = allocateIntermediateBuffers(B, S, D_out, H, HD);

    // Execute attention mechanism in stages
    computeQKVProjection(cmdBuff, input, tensors, W_q, W_k, W_v, B, S, D_in, D_out);
    computeAttentionScores(cmdBuff, tensors, B, H, S, HD);
    applyCausalMaskToScores(cmdBuff, tensors, B, H, S);
    computeSoftmax(cmdBuff, tensors, B, H, S);
    computeWeightedSum(cmdBuff, tensors, B, H, S, HD);
    combineHeadsAndProject(cmdBuff, tensors, W_out, output, B, S, D_out, H, HD);
}

// ============================================================================
// Private Helper Functions for MultiHeadAttentionNode
// ============================================================================

MultiHeadAttentionNode::IntermediateTensors
MultiHeadAttentionNode::allocateIntermediateBuffers(uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD)
{
    BufferPool& pool = BufferPool::get();
    IntermediateTensors tensors;

    tensors.Q_flat = Tensor(B, S, D);
    tensors.K_flat = Tensor(B, S, D);
    tensors.V_flat = Tensor(B, S, D);
    tensors.Q_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));
    tensors.K_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));
    tensors.V_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));

    tensors.scores = Tensor(B, H, S, S);
    tensors.scores.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*S*sizeof(float)));

    tensors.attn_weights = Tensor(B, H, S, S);
    tensors.attn_weights.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*S*sizeof(float)));

    tensors.context = Tensor(B, H, S, HD);
    tensors.context.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*HD*sizeof(float)));

    tensors.context_combined = Tensor(B, S, D);
    tensors.context_combined.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));

    return tensors;
}

void MultiHeadAttentionNode::computeQKVProjection(CommandBuffer& cmdBuff, const Tensor& input, IntermediateTensors& tensors,
                                                   const Tensor& W_q, const Tensor& W_k, const Tensor& W_v,
                                                   uint32_t B, uint32_t S, uint32_t D_in, uint32_t D_out)
{
    qkvProjDescSet.write({
        tensors.Q_flat.buffer(), tensors.K_flat.buffer(), tensors.V_flat.buffer(),
        input.buffer(),
        W_q.buffer(), W_k.buffer(), W_v.buffer()
    });

    int constants[] = {(int)B, (int)S, (int)D_in, (int)D_out};

    cmdBuff
        .bindPipeline(qkvProjection)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({qkvProjDescSet})
        .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D_out, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.Q_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.K_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.V_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeAttentionScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                      uint32_t B, uint32_t H, uint32_t S, uint32_t HD)
{
    scoresDescSet.write({
        tensors.scores.buffer(),
        tensors.Q_flat.buffer(),
        tensors.K_flat.buffer()
    });

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    struct { int B, H, S, HD; float scale; } constants = {(int)B, (int)H, (int)S, (int)HD, scale};

    cmdBuff
        .bindPipeline(attentionScores)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({scoresDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::applyCausalMaskToScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                       uint32_t B, uint32_t H, uint32_t S)
{
    maskDescSet.write({tensors.scores.buffer()});

    int constants[] = {(int)B, (int)H, (int)S};

    cmdBuff
        .bindPipeline(applyCausalMask)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({maskDescSet})
        .dispatch0(CEIL_DIV(B*H*S*S, 256))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeSoftmax(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                              uint32_t B, uint32_t H, uint32_t S)
{
    softmaxDescSet.write({
        tensors.attn_weights.buffer(),
        tensors.scores.buffer()
    });

    int constants[] = {(int)(B * H * S), (int)S};

    cmdBuff
        .bindPipeline(softmaxPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(B * H * S, 64))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.attn_weights.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeWeightedSum(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                  uint32_t B, uint32_t H, uint32_t S, uint32_t HD)
{
    weightedSumDescSet.write({
        tensors.context.buffer(),
        tensors.attn_weights.buffer(),
        tensors.V_flat.buffer()
    });

    int constants[] = {(int)B, (int)H, (int)S, (int)HD};

    cmdBuff
        .bindPipeline(weightedSum)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({weightedSumDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::combineHeadsAndProject(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                      const Tensor& W_out, Tensor& output,
                                                      uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD)
{
    // Combine heads
    combineDescSet.write({
        tensors.context_combined.buffer(),
        tensors.context.buffer()
    });

    int constants1[] = {(int)B, (int)H, (int)S, (int)HD};

    cmdBuff
        .bindPipeline(combineHeads)
        .setPushConstants(0, sizeof(constants1), constants1)
        .bindDescSets({combineDescSet})
        .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context_combined.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));

    // Output projection
    ComputePipeline outProjPipeline = requestPipeline(src_linear);
    DescriptorSet outProjDescSet = outProjPipeline.descSetLayout(0).newDescSet(gDestSetPool);

    outProjDescSet.write({
        output.buffer(),
        tensors.context_combined.buffer(),
        W_out.buffer()
    });

    int constants2[] = {(int)B, (int)S, (int)D, (int)d_out};

    cmdBuff
        .bindPipeline(outProjPipeline)
        .setPushConstants(0, sizeof(constants2), constants2)
        .bindDescSets({outProjDescSet})
        .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(d_out, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / output.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}
