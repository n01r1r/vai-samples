void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();
void transformerNodeTest();
void gpt2Test();

int main()
{
    // Run tokenizer tests
    tokenizerTest();

    // Run dataloader tests
    dataLoaderTest();

    // Run embedding node tests (Vulkan version)
    embeddingNodeTest();

    // Run attention node tests (Multi-Head Attention)
    attentionNodeTest();

    // Run transformer node tests (LayerNorm, GELU, FeedForward)
    transformerNodeTest();

    // Run GPT-2 model tests
    gpt2Test();

    return 0;
}

