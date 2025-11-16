#include "transformerNode.h"
#include "../attention/attentionNode.h"
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
// LayerNormNode: output = scale * (x - mean) / sqrt(var + eps) + shift
// ============================================================================

static const char* src_layer_norm = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // [B*S*D]
layout(set = 0, binding = 1) buffer Input { float x[]; };       // [B*S*D]
layout(set = 0, binding = 2) buffer Scale { float scale[]; };   // [D]
layout(set = 0, binding = 3) buffer Shift { float shift[]; };   // [D]

layout(push_constant) uniform PushConstants {
    int num_rows;  // B * S
    int D;         // d_model
    float eps;
};

void main() {
    int row = int(gl_GlobalInvocationID.x);
    if (row >= num_rows) return;

    int offset = row * D;

    // Compute mean
    float sum = 0.0;
    for (int i = 0; i < D; ++i) {
        sum += x[offset + i];
    }
    float mean = sum / float(D);

    // Compute variance
    float var_sum = 0.0;
    for (int i = 0; i < D; ++i) {
        float diff = x[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(D);

    // Normalize and apply scale/shift
    float inv_std = 1.0 / sqrt(variance + eps);
    for (int i = 0; i < D; ++i) {
        float norm_val = (x[offset + i] - mean) * inv_std;
        y[offset + i] = scale[i] * norm_val + shift[i];
    }
}
)";

LayerNormNode::LayerNormNode(uint32_t normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps)
{
    addSlot("in0", NodeSlot::input);
    addSlot("scale", NodeSlot::internal);  // Learnable parameter
    addSlot("shift", NodeSlot::internal);  // Learnable parameter
    addSlot("out0", NodeSlot::output);

    layerNormPipeline = requestPipeline(src_layer_norm);
    layerNormDescSet = layerNormPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void LayerNormNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == normalized_shape);

    // Initialize scale and shift if not set
    Tensor& scale = (*this)["scale"];
    Tensor& shift = (*this)["shift"];

    if (!scale.validShape()) {
        scale = Tensor(normalized_shape);
        // Initialize scale to 1.0
        std::vector<float> scale_data(normalized_shape, 1.0f);
        scale.set(scale_data);
    }

    if (!shift.validShape()) {
        shift = Tensor(normalized_shape);
        // Initialize shift to 0.0
        std::vector<float> shift_data(normalized_shape, 0.0f);
        shift.set(shift_data);
    }

    (*this)["out0"] = Tensor(B, S, D);
}

void LayerNormNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& scale = (*this)["scale"];
    Tensor& shift = (*this)["shift"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    uint32_t num_rows = B * S;

    layerNormDescSet.write({
        output.buffer(),
        input.buffer(),
        scale.buffer(),
        shift.buffer()
    });

    struct { int num_rows, D; float eps; } constants = {(int)num_rows, (int)D, eps};

    cmdBuff
        .bindPipeline(layerNormPipeline)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({layerNormDescSet})
        .dispatch0(CEIL_DIV(num_rows, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// GELUNode: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

static const char* src_gelu = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };
layout(set = 0, binding = 1) buffer Input { float x[]; };

layout(push_constant) uniform PushConstants {
    int N;  // Total number of elements
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N) return;

    float val = x[idx];

    // GELU approximation using tanh
    // sqrt(2/pi) ≈ 0.7978845608
    const float sqrt_2_over_pi = 0.7978845608;
    const float coeff = 0.044715;

    float inner = sqrt_2_over_pi * (val + coeff * val * val * val);
    float gelu = 0.5 * val * (1.0 + tanh(inner));

    y[idx] = gelu;
}
)";

GELUNode::GELUNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    geluPipeline = requestPipeline(src_gelu);
    geluDescSet = geluPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void GELUNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());

    // Output has same shape as input
    (*this)["out0"] = Tensor(input.shape());
}

void GELUNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];

    uint32_t N = input.numElements();

    geluDescSet.write({
        output.buffer(),
        input.buffer()
    });

    int constants[] = {(int)N};

    cmdBuff
        .bindPipeline(geluPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({geluDescSet})
        .dispatch0(CEIL_DIV(N, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// FeedForwardNode: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
// ============================================================================

static const char* src_linear_ff = R"(
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

FeedForwardNode::FeedForwardNode(uint32_t d_model)
    : d_model(d_model), hidden_dim(4 * d_model)
{
    addSlot("in0", NodeSlot::input);
    addSlot("weight1", NodeSlot::internal);  // [4*d_model, d_model]
    addSlot("weight2", NodeSlot::internal);  // [d_model, 4*d_model]
    addSlot("out0", NodeSlot::output);

    linear1Pipeline = requestPipeline(src_linear_ff);
    geluPipeline = requestPipeline(src_gelu);
    linear2Pipeline = requestPipeline(src_linear_ff);

    linear1DescSet = linear1Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    geluDescSet = geluPipeline.descSetLayout(0).newDescSet(gDestSetPool);
    linear2DescSet = linear2Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void FeedForwardNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == d_model);

    // Initialize weights if not set
    Tensor& weight1 = (*this)["weight1"];
    Tensor& weight2 = (*this)["weight2"];

    if (!weight1.validShape()) {
        weight1 = Tensor(hidden_dim, d_model);
    }

    if (!weight2.validShape()) {
        weight2 = Tensor(d_model, hidden_dim);
    }

    (*this)["out0"] = Tensor(B, S, D);
}

void FeedForwardNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& weight1 = (*this)["weight1"];
    Tensor& weight2 = (*this)["weight2"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = d_model;
    uint32_t H = hidden_dim;

    // Allocate temporary buffers
    BufferPool& pool = BufferPool::get();

    Tensor hidden(B, S, H);
    hidden.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*H*sizeof(float)));

    Tensor gelu_out(B, S, H);
    gelu_out.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*H*sizeof(float)));

    // Step 1: Linear1 (d_model -> 4*d_model)
    {
        linear1DescSet.write({
            hidden.buffer(),
            input.buffer(),
            weight1.buffer()
        });

        int constants[] = {(int)B, (int)S, (int)D, (int)H};

        cmdBuff
            .bindPipeline(linear1Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear1DescSet})
            .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(H, 16))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / hidden.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }

    // Step 2: GELU activation
    {
        geluDescSet.write({
            gelu_out.buffer(),
            hidden.buffer()
        });

        int constants[] = {(int)(B * S * H)};

        cmdBuff
            .bindPipeline(geluPipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({geluDescSet})
            .dispatch0(CEIL_DIV(B * S * H, 256))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / gelu_out.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }

    // Step 3: Linear2 (4*d_model -> d_model)
    {
        linear2DescSet.write({
            output.buffer(),
            gelu_out.buffer(),
            weight2.buffer()
        });

        int constants[] = {(int)B, (int)S, (int)H, (int)D};

        cmdBuff
            .bindPipeline(linear2Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear2DescSet})
            .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D, 16))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / output.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }
}

// ============================================================================
// TransformerBlockNode: Pre-norm architecture with residual connections
// ============================================================================

static const char* src_residual_add = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // output
layout(set = 0, binding = 1) buffer Residual { float res[]; };  // residual
layout(set = 0, binding = 2) buffer Input { float x[]; };       // input

layout(push_constant) uniform PushConstants {
    int N;  // Total number of elements
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N) return;

    y[idx] = res[idx] + x[idx];
}
)";

TransformerBlockNode::TransformerBlockNode(uint32_t d_model, uint32_t num_heads)
    : d_model(d_model), num_heads(num_heads)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    // Weights for LayerNorm1
    addSlot("norm1_scale", NodeSlot::internal);
    addSlot("norm1_shift", NodeSlot::internal);

    // Weights for MultiHeadAttention
    addSlot("attn_wq", NodeSlot::internal);
    addSlot("attn_wk", NodeSlot::internal);
    addSlot("attn_wv", NodeSlot::internal);
    addSlot("attn_wout", NodeSlot::internal);

    // Weights for LayerNorm2
    addSlot("norm2_scale", NodeSlot::internal);
    addSlot("norm2_shift", NodeSlot::internal);

    // Weights for FeedForward
    addSlot("ff_w1", NodeSlot::internal);
    addSlot("ff_w2", NodeSlot::internal);

    norm1Pipeline = requestPipeline(src_layer_norm);
    norm2Pipeline = requestPipeline(src_layer_norm);
    residualAdd1Pipeline = requestPipeline(src_residual_add);
    residualAdd2Pipeline = requestPipeline(src_residual_add);

    norm1DescSet = norm1Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    norm2DescSet = norm2Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    residualAdd1DescSet = residualAdd1Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    residualAdd2DescSet = residualAdd2Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void TransformerBlockNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == d_model);

    // Initialize LayerNorm1 weights if not set
    Tensor& norm1_scale = (*this)["norm1_scale"];
    Tensor& norm1_shift = (*this)["norm1_shift"];
    if (!norm1_scale.validShape()) {
        norm1_scale = Tensor(d_model);
        std::vector<float> scale_data(d_model, 1.0f);
        norm1_scale.set(scale_data);
    }
    if (!norm1_shift.validShape()) {
        norm1_shift = Tensor(d_model);
        std::vector<float> shift_data(d_model, 0.0f);
        norm1_shift.set(shift_data);
    }

    // Initialize Attention weights if not set
    Tensor& attn_wq = (*this)["attn_wq"];
    Tensor& attn_wk = (*this)["attn_wk"];
    Tensor& attn_wv = (*this)["attn_wv"];
    Tensor& attn_wout = (*this)["attn_wout"];
    if (!attn_wq.validShape()) attn_wq = Tensor(d_model, d_model);
    if (!attn_wk.validShape()) attn_wk = Tensor(d_model, d_model);
    if (!attn_wv.validShape()) attn_wv = Tensor(d_model, d_model);
    if (!attn_wout.validShape()) attn_wout = Tensor(d_model, d_model);

    // Initialize LayerNorm2 weights if not set
    Tensor& norm2_scale = (*this)["norm2_scale"];
    Tensor& norm2_shift = (*this)["norm2_shift"];
    if (!norm2_scale.validShape()) {
        norm2_scale = Tensor(d_model);
        std::vector<float> scale_data(d_model, 1.0f);
        norm2_scale.set(scale_data);
    }
    if (!norm2_shift.validShape()) {
        norm2_shift = Tensor(d_model);
        std::vector<float> shift_data(d_model, 0.0f);
        norm2_shift.set(shift_data);
    }

    // Initialize FeedForward weights if not set
    Tensor& ff_w1 = (*this)["ff_w1"];
    Tensor& ff_w2 = (*this)["ff_w2"];
    if (!ff_w1.validShape()) ff_w1 = Tensor(4 * d_model, d_model);
    if (!ff_w2.validShape()) ff_w2 = Tensor(d_model, 4 * d_model);

    (*this)["out0"] = Tensor(B, S, D);
}

void TransformerBlockNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = d_model;
    uint32_t num_rows = B * S;

    // Get weights
    Tensor& norm1_scale = (*this)["norm1_scale"];
    Tensor& norm1_shift = (*this)["norm1_shift"];
    Tensor& attn_wq = (*this)["attn_wq"];
    Tensor& attn_wk = (*this)["attn_wk"];
    Tensor& attn_wv = (*this)["attn_wv"];
    Tensor& attn_wout = (*this)["attn_wout"];
    Tensor& norm2_scale = (*this)["norm2_scale"];
    Tensor& norm2_shift = (*this)["norm2_shift"];
    Tensor& ff_w1 = (*this)["ff_w1"];
    Tensor& ff_w2 = (*this)["ff_w2"];

    // Allocate temporary buffers
    Tensor norm1_out = Tensor(B, S, D);
    norm1_out.bindBuffer(netGlobalDevice.createBuffer({
        .size = B * S * D * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    }));

    Tensor attn_out = Tensor(B, S, D);
    attn_out.bindBuffer(netGlobalDevice.createBuffer({
        .size = B * S * D * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    }));

    Tensor residual1 = Tensor(B, S, D);
    residual1.bindBuffer(netGlobalDevice.createBuffer({
        .size = B * S * D * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    }));

    Tensor norm2_out = Tensor(B, S, D);
    norm2_out.bindBuffer(netGlobalDevice.createBuffer({
        .size = B * S * D * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    }));

    Tensor ff_out = Tensor(B, S, D);
    ff_out.bindBuffer(netGlobalDevice.createBuffer({
        .size = B * S * D * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    }));

    // Step 1: LayerNorm1(input) -> norm1_out
    {
        norm1DescSet.write({
            norm1_out.buffer(),
            input.buffer(),
            norm1_scale.buffer(),
            norm1_shift.buffer()
        });

        struct { int num_rows, D; float eps; } constants = {(int)num_rows, (int)D, 1e-5f};

        cmdBuff
            .bindPipeline(norm1Pipeline)
            .setPushConstants(0, sizeof(constants), &constants)
            .bindDescSets({norm1DescSet})
            .dispatch0(CEIL_DIV(num_rows, 256))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / norm1_out.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }

    // Step 2: MultiHeadAttention(norm1_out) -> attn_out
    {
        // Create temporary attention node
        MultiHeadAttentionNode attn(d_model, d_model, num_heads);
        attn["in0"] = norm1_out;
        attn["W_query"] = attn_wq;
        attn["W_key"] = attn_wk;
        attn["W_value"] = attn_wv;
        attn["W_out"] = attn_wout;
        attn.prepare();
        attn["out0"] = attn_out;  // Assign our pre-allocated buffer AFTER prepare()
        attn.run(cmdBuff);
    }

    // Step 3: residual1 = input + attn_out
    {
        residualAdd1DescSet.write({
            residual1.buffer(),
            input.buffer(),
            attn_out.buffer()
        });

        int constants[] = {(int)(B * S * D)};

        cmdBuff
            .bindPipeline(residualAdd1Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({residualAdd1DescSet})
            .dispatch0(CEIL_DIV(B * S * D, 256))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / residual1.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }

    // Step 4: LayerNorm2(residual1) -> norm2_out
    {
        norm2DescSet.write({
            norm2_out.buffer(),
            residual1.buffer(),
            norm2_scale.buffer(),
            norm2_shift.buffer()
        });

        struct { int num_rows, D; float eps; } constants = {(int)num_rows, (int)D, 1e-5f};

        cmdBuff
            .bindPipeline(norm2Pipeline)
            .setPushConstants(0, sizeof(constants), &constants)
            .bindDescSets({norm2DescSet})
            .dispatch0(CEIL_DIV(num_rows, 256))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / norm2_out.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }

    // Step 5: FeedForward(norm2_out) -> ff_out
    {
        // Create temporary feedforward node
        FeedForwardNode ff(d_model);
        ff["in0"] = norm2_out;
        ff["weight1"] = ff_w1;
        ff["weight2"] = ff_w2;
        ff.prepare();
        ff["out0"] = ff_out;  // Assign our pre-allocated buffer AFTER prepare()
        ff.run(cmdBuff);
    }

    // Step 6: output = residual1 + ff_out
    {
        residualAdd2DescSet.write({
            output.buffer(),
            residual1.buffer(),
            ff_out.buffer()
        });

        int constants[] = {(int)(B * S * D)};

        cmdBuff
            .bindPipeline(residualAdd2Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({residualAdd2DescSet})
            .dispatch0(CEIL_DIV(B * S * D, 256))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / output.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }
}

