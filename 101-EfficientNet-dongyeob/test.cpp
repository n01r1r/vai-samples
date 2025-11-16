#include "neuralNet.h"
#include "neuralNodes.h"
#include "efficientNet.h"
#include "jsonParser.h"
#include <stb/stb_image.h>
#include <cstring>  // memcpy


template<uint32_t Channels>
auto readImage(const char* filename)
{
    int w, h, c0, c = Channels;
    std::vector<uint8_t> srcImage;

    if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
    {
        srcImage.assign(input, input + w * h * c);
        stbi_image_free(input);
    }
    else
    {
        printf(stbi_failure_reason());
        fflush(stdout);
        throw;
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}


Tensor eval_efficientnet(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter)
{
    // TODO: Create EfficientNet network
    // TODO: Load weights from JSON
    // TODO: Create input tensor
    // TODO: Run inference
    Tensor result;
    return result;
}


void test()
{
    void loadShaders();
    loadShaders();

    // TODO: Load test image
    // TODO: Preprocess image (normalize, etc.)
    // TODO: Load weights
    // TODO: Run evaluation
    // TODO: Copy results and display
}

