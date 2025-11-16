#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy




class TNetBlock : public NodeGroup
{
    uint32_t K;

    PointWiseMLPNode mlp[3];
    MaxPooling1DNode maxpool;
    FullyConnectedNode fc[3];

public:
    TNetBlock(uint32_t inputDim)
    : K(inputDim),
    mlp{PointWiseMLPNode(K, 64), PointWiseMLPNode(64, 128), PointWiseMLPNode(128, 1024)},
    maxpool(MaxPooling1DNode()),
    fc{FullyConnectedNode(1024, 512), FullyConnectedNode(512, 256), FullyConnectedNode(256, K*K)}
    {
        mlp[0] - mlp[1] - mlp[2] - maxpool - fc[0] - fc[1] - fc[2];
        defineSlot("in0", mlp[0].slot("in0"));
        defineSlot("out0", fc[2].slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        // mlpX.* e.g. mlp0.weight, mlp0.bias, mlp1.weight, mlp1.bias, mlp2.weight, mlp2.bias
        for (int i = 0; i < 3; ++i)
        {
            const std::string prefix = "mlp" + std::to_string(i) + ".";
            if (name.starts_with(prefix))
                return mlp[i][name.substr(prefix.length())];
        }

        // fcX.* e.g. fc0.weight, fc0.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias
        for (int i = 0; i < 3; ++i)
        {
            const std::string prefix = "fc" + std::to_string(i) + ".";
            if (name.starts_with(prefix))
                return fc[i][name.substr(prefix.length())];
        }

        throw std::runtime_error("No such layer in TNetBlock: " + name);
    }
};

template<uint32_t nBlocks>
class ConvSequence : public NodeGroup
{
    uint32_t K;
    uint32_t C[nBlocks + 1];
    std::unique_ptr<ConvBlock> blocks[nBlocks];
    
public:
    ConvSequence(const uint32_t(&channels)[nBlocks + 1], uint32_t kernel)
    : K(kernel)
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            C[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<ConvBlock>(C[i], C[i + 1], K);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
            if (name.starts_with("conv" + std::to_string(i) + "."))
                return (*blocks[i])[name.substr(6)];
        throw std::runtime_error("No such layer in ConvSequence: " + name);
    }
};
template<std::size_t N>
ConvSequence(const uint32_t (&)[N], uint32_t) -> ConvSequence<N - 1>;


class MnistNet : public NeuralNet
{
    ConvSequence<2> convX2;
    FlattenNode flatten;
    FullyConnectedNode fc;

public:
    MnistNet(Device& device)
    : NeuralNet(device, 1, 1)
    , convX2({1, 32, 64}, 3)
    , fc(7 * 7 * 64, 10)
    {
        input(0) - convX2 - flatten - fc - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("conv"))
            return convX2[name];  // conv0, conv1 등을 convX2에 전달
        else if (name == "weight" || name == "bias")
            return fc[name];  // fc의 weight와 bias
        else if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in MnistNet: " + name);
    }
};


Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    MnistNet mnistNet(netGlobalDevice);

    mnistNet["conv0.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);
    mnistNet["conv0.bias"] = Tensor(json["layer1.0.bias"]);
    mnistNet["conv1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
    mnistNet["conv1.bias"] = Tensor(json["layer2.0.bias"]);
    mnistNet["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);
    mnistNet["bias"] = Tensor(json["fc.bias"]);
    
    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = mnistNet(inputTensor)[0];

    return result;
}

void test()
{
    void loadShaders();
    loadShaders();

    const uint32_t channels = 1;
    auto [srcImage, width, height] = readImage<channels>(PROJECT_ROOT_DIR"/data/0.png");
    _ASSERT(width == 28 && height == 28);
    _ASSERT(width * height * channels == srcImage.size());

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");

    uint32_t iter = 1;
    Tensor eval;

    {
        TimeChecker timer("(VAI) MNIST evaluation: {} iterations", iter);
        eval = eval_mnist(inputData, json, iter);
    }

    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        10 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    vk::Buffer evalBuffer = eval.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, evalBuffer)
        .end()
        .submit()
        .wait();

    float data[10];
    memcpy(data, outBuffer.map(), 10 * sizeof(float));

    for(int i=0; i<10; ++i)
        printf("data[%d] = %f\n", i, data[i]);
}





// Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
// {
//     NeuralNet mnistNet(netGlobalDevice); 

//     auto conv1 = ConvolutionNode(1, 32, 3); 
//     auto relu1 = ReluNode();
//     auto maxpool1 = MaxPoolingNode(2);
//     auto conv2 = ConvolutionNode(32, 64, 3);
//     auto relu2 = ReluNode();
//     auto maxpool2 = MaxPoolingNode(2);
//     auto flatten = FlattenNode();
//     auto fc = FullyConnectedNode(7*7*64, 10); 

//     mnistNet.input(0) - conv1 - relu1 - maxpool1 - conv2 - relu2 - maxpool2 - flatten - fc - mnistNet.output(0);

//     conv1["weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);                     // 32 x 1 x 3 x 3 => 1*3*3 x 32
//     conv1["bias"] = Tensor(json["layer1.0.bias"]);                                                          // 32
//     conv2["weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);                    // 64 x 32 x 3 x 3 => 32*3*3 x 64
//     conv2["bias"] = Tensor(json["layer2.0.bias"]);                                                          // 64                                        
//     fc["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);     // 10 x 64*7*7 => 7*7*64 x 10
//     fc["bias"] = Tensor(json["fc.bias"]);                                                                   // 10
    
//     Tensor result;
//     Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

//     for (uint32_t i = 0; i < iter; ++i)
//         result = mnistNet(inputTensor)[0];

//     return result;
// }

