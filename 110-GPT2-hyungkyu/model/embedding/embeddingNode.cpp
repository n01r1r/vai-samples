#include "embeddingNode.h"
#include "../../core/error.h"
#include <unordered_map>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

using namespace vk;

// ==================== GLSL Shaders ====================

// Token Embedding Lookup Shader
static const char* src_token_embedding = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };      // [B*S*E]
layout(set = 0, binding = 1) buffer InBuffer { float token_ids[]; };   // [B*S] (as float)
layout(set = 0, binding = 2) buffer WeightBuffer { float weight[]; };  // [V*E]
layout(push_constant) uniform PushConstants {
    int BS;  // batch_size * seq_length
    int V;   // vocab_size
    int E;   // embedding_dim
};

void main()
{
    int bs = int(gl_GlobalInvocationID.x);  // batch*seq index
    if (bs >= BS) return;

    int token_id = int(token_ids[bs]);

    // Copy embedding from weight table
    for (int e = 0; e < E; ++e) {
        out0[bs * E + e] = weight[token_id * E + e];
    }
})";


// Positional Embedding Shader
static const char* src_positional_embedding = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };      // [B*S*E]
layout(set = 0, binding = 1) buffer WeightBuffer { float weight[]; };  // [M*E]
layout(push_constant) uniform PushConstants {
    int B;   // batch_size
    int S;   // seq_length
    int M;   // max_length
    int E;   // embedding_dim
};

void main()
{
    int bs = int(gl_GlobalInvocationID.x);  // batch*seq index
    int BS = B * S;
    if (bs >= BS) return;

    int s = bs % S;  // position in sequence

    // Copy positional embedding from weight table
    for (int e = 0; e < E; ++e) {
        out0[bs * E + e] = weight[s * E + e];
    }
})";


// Add Embeddings Shader
static const char* src_add_embeddings = R"(
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };       // [B*S*E]
layout(set = 0, binding = 1) buffer TokenEmb { float token_emb[]; };   // [B*S*E]
layout(set = 0, binding = 2) buffer PosEmb { float pos_emb[]; };       // [B*S*E]
layout(push_constant) uniform PushConstants {
    int BSE;  // batch_size * seq_length * embedding_dim
};

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= BSE) return;

    out0[idx] = token_emb[idx] + pos_emb[idx];
})";


// ==================== Global Variables ====================

extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}


// ==================== TokenEmbeddingNode ====================

TokenEmbeddingNode::TokenEmbeddingNode(uint32_t vocab_size, uint32_t embedding_dim)
    : V(vocab_size), E(embedding_dim)
{
    addSlot("in0", NodeSlot::input);      // token_ids [B, S]
    addSlot("weight", NodeSlot::internal); // embedding table [V, E]
    addSlot("out0", NodeSlot::output);     // embeddings [B, S, E]

    tokenEmbedding = requestPipeline(src_token_embedding);
    tokenEmbeddingDescSet = tokenEmbedding.descSetLayout(0).newDescSet(gDestSetPool);
}

void TokenEmbeddingNode::prepare()
{
    Tensor& tokenIds = (*this)["in0"];
    ASSERT_(tokenIds.validShape());
    ASSERT_(tokenIds.shape().size() == 2);

    uint32_t B = tokenIds.shape()[0];  // batch_size
    uint32_t S = tokenIds.shape()[1];  // seq_length

    // Weight: embedding table [V, E]
    Tensor& weight = (*this)["weight"];
    if (!weight.validShape()) {
        weight = Tensor(V, E);
    }

    // Output: embeddings [B, S, E]
    (*this)["out0"] = Tensor(B, S, E);
}

void TokenEmbeddingNode::run(CommandBuffer cmdBuff)
{
    Tensor& tokenIds = (*this)["in0"];
    Tensor& weight = (*this)["weight"];
    Tensor& out = (*this)["out0"];

    uint32_t B = tokenIds.shape()[0];
    uint32_t S = tokenIds.shape()[1];
    uint32_t BS = B * S;

    tokenEmbeddingDescSet.write({
        out.buffer(),
        tokenIds.buffer(),
        weight.buffer()
    });

    int constants[] = {(int)BS, (int)V, (int)E};

    cmdBuff
        .bindPipeline(tokenEmbedding)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({tokenEmbeddingDescSet})
        .dispatch0(CEIL_DIV(BS, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


// ==================== PositionalEmbeddingNode ====================

PositionalEmbeddingNode::PositionalEmbeddingNode(uint32_t max_length, uint32_t embedding_dim)
    : M(max_length), E(embedding_dim)
{
    addSlot("in0", NodeSlot::input);       // dummy input to get shape [B, S]
    addSlot("weight", NodeSlot::internal);  // pos embedding table [M, E]
    addSlot("out0", NodeSlot::output);      // pos embeddings [B, S, E]

    positionalEmbedding = requestPipeline(src_positional_embedding);
    positionalEmbeddingDescSet = positionalEmbedding.descSetLayout(0).newDescSet(gDestSetPool);
}

void PositionalEmbeddingNode::prepare()
{
    Tensor& input = (*this)["in0"];
    ASSERT_(input.validShape());
    ASSERT_(input.shape().size() == 2);

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];

    // Weight: positional embedding table [M, E]
    Tensor& weight = (*this)["weight"];
    if (!weight.validShape()) {
        weight = Tensor(M, E);
    }

    // Output: positional embeddings [B, S, E]
    (*this)["out0"] = Tensor(B, S, E);
}

void PositionalEmbeddingNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& weight = (*this)["weight"];
    Tensor& out = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t BS = B * S;

    positionalEmbeddingDescSet.write({
        out.buffer(),
        weight.buffer()
    });

    int constants[] = {(int)B, (int)S, (int)M, (int)E};

    cmdBuff
        .bindPipeline(positionalEmbedding)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({positionalEmbeddingDescSet})
        .dispatch0(CEIL_DIV(BS, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


// ==================== GPTEmbeddingNode ====================

GPTEmbeddingNode::GPTEmbeddingNode(uint32_t vocab_size, uint32_t max_length, uint32_t embedding_dim)
    : V(vocab_size), M(max_length), E(embedding_dim)
{
    addSlot("in0", NodeSlot::input);               // token_ids [B, S]
    addSlot("token_weight", NodeSlot::internal);   // token embedding table [V, E]
    addSlot("pos_weight", NodeSlot::internal);     // pos embedding table [M, E]
    addSlot("out0", NodeSlot::output);             // combined embeddings [B, S, E]

    tokenEmbedding = requestPipeline(src_token_embedding);
    positionalEmbedding = requestPipeline(src_positional_embedding);
    addEmbeddings = requestPipeline(src_add_embeddings);

    tokenEmbeddingDescSet = tokenEmbedding.descSetLayout(0).newDescSet(gDestSetPool);
    positionalEmbeddingDescSet = positionalEmbedding.descSetLayout(0).newDescSet(gDestSetPool);
    addEmbeddingsDescSet = addEmbeddings.descSetLayout(0).newDescSet(gDestSetPool);
}

void GPTEmbeddingNode::prepare()
{
    Tensor& tokenIds = (*this)["in0"];
    ASSERT_(tokenIds.validShape());
    ASSERT_(tokenIds.shape().size() == 2);

    uint32_t B = tokenIds.shape()[0];
    uint32_t S = tokenIds.shape()[1];

    // Initialize weight tables if needed
    Tensor& tokenWeight = (*this)["token_weight"];
    if (!tokenWeight.validShape()) {
        tokenWeight = Tensor(V, E);
    }

    Tensor& posWeight = (*this)["pos_weight"];
    if (!posWeight.validShape()) {
        posWeight = Tensor(M, E);
    }

    // Output
    (*this)["out0"] = Tensor(B, S, E);
}

void GPTEmbeddingNode::run(CommandBuffer cmdBuff)
{
    Tensor& tokenIds = (*this)["in0"];
    Tensor& tokenWeight = (*this)["token_weight"];
    Tensor& posWeight = (*this)["pos_weight"];
    Tensor& out = (*this)["out0"];

    uint32_t B = tokenIds.shape()[0];
    uint32_t S = tokenIds.shape()[1];
    uint32_t BS = B * S;
    uint32_t BSE = BS * E;

    // Allocate temporary buffers for intermediate results
    BufferPool& pool = BufferPool::get();
    Buffer tokenEmbBuffer = pool.requestBuffer(
        netGlobalDevice,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        BS * E * sizeof(float)
    );

    Buffer posEmbBuffer = pool.requestBuffer(
        netGlobalDevice,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        BS * E * sizeof(float)
    );

    // Step 1: Token Embedding
    tokenEmbeddingDescSet.write({
        tokenEmbBuffer,
        tokenIds.buffer(),
        tokenWeight.buffer()
    });

    int tokenConstants[] = {(int)BS, (int)V, (int)E};

    cmdBuff
        .bindPipeline(tokenEmbedding)
        .setPushConstants(0, sizeof(tokenConstants), tokenConstants)
        .bindDescSets({tokenEmbeddingDescSet})
        .dispatch0(CEIL_DIV(BS, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / tokenEmbBuffer
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Step 2: Positional Embedding
    positionalEmbeddingDescSet.write({
        posEmbBuffer,
        posWeight.buffer()
    });

    int posConstants[] = {(int)B, (int)S, (int)M, (int)E};

    cmdBuff
        .bindPipeline(positionalEmbedding)
        .setPushConstants(0, sizeof(posConstants), posConstants)
        .bindDescSets({positionalEmbeddingDescSet})
        .dispatch0(CEIL_DIV(BS, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / posEmbBuffer
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Step 3: Add Embeddings
    addEmbeddingsDescSet.write({
        out.buffer(),
        tokenEmbBuffer,
        posEmbBuffer
    });

    cmdBuff
        .bindPipeline(addEmbeddings)
        .setPushConstants(0, sizeof(int), &BSE)
        .bindDescSets({addEmbeddingsDescSet})
        .dispatch0(CEIL_DIV(BSE, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Return temporary buffers to pool
    pool.returnBuffer(tokenEmbBuffer);
    pool.returnBuffer(posEmbBuffer);
}
