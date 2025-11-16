#include "gpt2.h"
#include "../core/error.h"
#include <cmath>
#include <unordered_map>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static ComputePipeline requestPipeline(Device& device, const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = device.createComputePipeline({src});
    return it->second;
}

// ============================================================================
// Language Modeling Head Shader: Y = X @ W^T
// Projects from d_model to vocab_size
// ============================================================================

static const char* src_lm_head = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };    // [B*S, V]
layout(set = 0, binding = 1) buffer Input { float x[]; };     // [B*S, D]
layout(set = 0, binding = 2) buffer Weight { float w[]; };    // [V, D]

layout(push_constant) uniform PushConstants {
    int BS;  // batch * seq_len
    int D;   // d_model
    int V;   // vocab_size
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int v = int(gl_GlobalInvocationID.y);

    if (bs >= BS || v >= V) return;

    // Y[bs, v] = sum_d(X[bs, d] * W[v, d])
    float sum = 0.0;
    for (int d = 0; d < D; ++d) {
        sum += x[bs * D + d] * w[v * D + d];
    }
    y[bs * V + v] = sum;
}
)";

// ============================================================================
// GPT-2 Configuration Presets
// ============================================================================

GPT2Config GPT2SmallConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 768,
        .num_heads = 12,
        .num_layers = 12,
        .dropout = 0.1f
    };
}

GPT2Config GPT2MediumConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1024,
        .num_heads = 16,
        .num_layers = 24,
        .dropout = 0.1f
    };
}

GPT2Config GPT2LargeConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1280,
        .num_heads = 20,
        .num_layers = 36,
        .dropout = 0.1f
    };
}

GPT2Config GPT2XLConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1600,
        .num_heads = 25,
        .num_layers = 48,
        .dropout = 0.1f
    };
}

// ============================================================================
// GPT-2 Implementation
// ============================================================================

GPT2::GPT2(Device& device, DescriptorPool& descPool, const GPT2Config& config)
    : config(config), device(device), descPool(descPool), net(device, 1, 1)
{
    buildModel();
    initializeWeights();
}

GPT2::~GPT2()
{
    delete embedding;
    for (auto* block : transformerBlocks) {
        delete block;
    }
    delete finalNorm;
}

void GPT2::buildModel()
{
    // 1. Embedding layer (token + positional)
    embedding = new GPTEmbeddingNode(config.vocab_size, config.max_seq_len, config.d_model);

    // 2. Transformer blocks
    transformerBlocks.reserve(config.num_layers);
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        transformerBlocks.push_back(new TransformerBlockNode(config.d_model, config.num_heads));
    }

    // 3. Final layer normalization
    finalNorm = new LayerNormNode(config.d_model);

    // 4. Build neural network graph
    // Input -> Embedding
    net.input(0) - *embedding;

    // Embedding -> TransformerBlocks (sequential)
    Node* prevNode = embedding;
    for (auto* block : transformerBlocks) {
        *prevNode - *block;
        prevNode = block;
    }

    // Last TransformerBlock -> Final LayerNorm -> Output
    *prevNode - *finalNorm - net.output(0);

    // 5. Create LM head pipeline (for projecting to vocabulary)
    lmHeadPipeline = requestPipeline(device, src_lm_head);
    lmHeadDescSet = lmHeadPipeline.descSetLayout(0).newDescSet(descPool);
}

void GPT2::initializeWeights()
{
    // Initialize embedding weights if not already set
    Tensor& token_emb = (*embedding)["token_weight"];
    Tensor& pos_emb = (*embedding)["pos_weight"];

    if (!token_emb.validShape()) {
        // Initialize with small random values (normally would load pretrained)
        std::vector<float> token_data(config.vocab_size * config.d_model, 0.01f);
        token_emb = Tensor(config.vocab_size, config.d_model).set(token_data);
    }
    if (!pos_emb.validShape()) {
        // Initialize positional embeddings with small values
        std::vector<float> pos_data(config.max_seq_len * config.d_model, 0.01f);
        pos_emb = Tensor(config.max_seq_len, config.d_model).set(pos_data);
    }

    // Initialize transformer block weights
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        auto* block = transformerBlocks[i];

        // LayerNorm1
        if (!(*block)["norm1_scale"].validShape()) {
            std::vector<float> scale_data(config.d_model, 1.0f);
            (*block)["norm1_scale"] = Tensor(config.d_model).set(scale_data);
        }
        if (!(*block)["norm1_shift"].validShape()) {
            std::vector<float> shift_data(config.d_model, 0.0f);
            (*block)["norm1_shift"] = Tensor(config.d_model).set(shift_data);
        }

        // Attention weights
        if (!(*block)["attn_wq"].validShape()) {
            std::vector<float> wq_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wq"] = Tensor(config.d_model, config.d_model).set(wq_data);
        }
        if (!(*block)["attn_wk"].validShape()) {
            std::vector<float> wk_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wk"] = Tensor(config.d_model, config.d_model).set(wk_data);
        }
        if (!(*block)["attn_wv"].validShape()) {
            std::vector<float> wv_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wv"] = Tensor(config.d_model, config.d_model).set(wv_data);
        }
        if (!(*block)["attn_wout"].validShape()) {
            std::vector<float> wout_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wout"] = Tensor(config.d_model, config.d_model).set(wout_data);
        }

        // LayerNorm2
        if (!(*block)["norm2_scale"].validShape()) {
            std::vector<float> scale_data(config.d_model, 1.0f);
            (*block)["norm2_scale"] = Tensor(config.d_model).set(scale_data);
        }
        if (!(*block)["norm2_shift"].validShape()) {
            std::vector<float> shift_data(config.d_model, 0.0f);
            (*block)["norm2_shift"] = Tensor(config.d_model).set(shift_data);
        }

        // FeedForward weights
        if (!(*block)["ff_w1"].validShape()) {
            std::vector<float> w1_data(4 * config.d_model * config.d_model, 0.01f);
            (*block)["ff_w1"] = Tensor(4 * config.d_model, config.d_model).set(w1_data);
        }
        if (!(*block)["ff_w2"].validShape()) {
            std::vector<float> w2_data(config.d_model * 4 * config.d_model, 0.01f);
            (*block)["ff_w2"] = Tensor(config.d_model, 4 * config.d_model).set(w2_data);
        }
    }

    // Initialize final LayerNorm weights
    if (!(*finalNorm)["scale"].validShape()) {
        std::vector<float> scale_data(config.d_model, 1.0f);
        (*finalNorm)["scale"] = Tensor(config.d_model).set(scale_data);
    }
    if (!(*finalNorm)["shift"].validShape()) {
        std::vector<float> shift_data(config.d_model, 0.0f);
        (*finalNorm)["shift"] = Tensor(config.d_model).set(shift_data);
    }
}

Tensor GPT2::forward(const Tensor& input_ids)
{
    _ASSERT(input_ids.shape().size() == 2);  // [batch, seq_len]

    uint32_t B = input_ids.shape()[0];
    uint32_t S = input_ids.shape()[1];
    _ASSERT(S <= config.max_seq_len);

    // Run through transformer layers
    std::vector<Tensor> outputs = net(input_ids);
    Tensor& normalized = outputs[0];  // [B, S, d_model]

    // For now, just return the normalized output from the transformer
    // The LM head projection can be added later or done separately
    // This is just to test that the transformer stack works
    return normalized;
}
