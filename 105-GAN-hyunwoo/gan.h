#ifndef GAN_H
#define GAN_H

#include "neuralNet.h"
#include "neuralNodes.h"
#include <random>

// Simple DCGAN (Deep Convolutional GAN) for MNIST
// Generator: [Noise 100] -> FC -> Reshape -> TransConv -> BatchNorm -> LeakyReLU -> ... -> Tanh -> [28x28x1]
// Discriminator: [28x28x1] -> Conv -> LeakyReLU -> ... -> FC -> Sigmoid -> [0 or 1]

class Generator : public NeuralNet
{
    // Architecture: Latent(100) -> FC(7*7*256) -> Reshape(7,7,256) ->
    //               TransConv(128) -> BN -> LeakyReLU ->
    //               TransConv(64) -> BN -> LeakyReLU ->
    //               TransConv(1) -> Tanh

    FullyConnectedNode fc;
    ReshapeNode reshape;

    TransposeConvNode transConv1;
    BatchNormNode bn1;
    LeakyReluNode lrelu1;

    TransposeConvNode transConv2;
    BatchNormNode bn2;
    LeakyReluNode lrelu2;

    TransposeConvNode transConv3;
    TanhNode tanh;

public:
    Generator(Device& device)
    : NeuralNet(device, 1, 1)
    , fc(100, 7*7*256)
    , reshape({7, 7, 256})
    , transConv1(256, 128, 5, 2)  // 7x7 -> 15x15 (changed from 4 to 5)
    , bn1(128)
    , lrelu1(0.2f)
    , transConv2(128, 64, 5, 2)   // 15x15 -> 31x31 (changed from 4 to 5)
    , bn2(64)
    , lrelu2(0.2f)
    , transConv3(64, 1, 3, 1)     // 31x31 -> 31x31 (output layer)
    {
        // Connect the graph
        input(0) - fc - reshape -
        transConv1 - bn1 - lrelu1 -
        transConv2 - bn2 - lrelu2 -
        transConv3 - tanh - output(0);
    }

    // Generate random noise as input
    Tensor generateNoise(uint32_t batchSize = 1)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> noise(batchSize * 100);
        for (auto& val : noise)
            val = dist(gen);

        return Tensor(batchSize, 100).set(noise);
    }

    // Access to weights (for loading/saving)
    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else if (name.starts_with("transConv1."))
            return transConv1[name.substr(11)];
        else if (name.starts_with("bn1."))
            return bn1[name.substr(4)];
        else if (name.starts_with("transConv2."))
            return transConv2[name.substr(11)];
        else if (name.starts_with("bn2."))
            return bn2[name.substr(4)];
        else if (name.starts_with("transConv3."))
            return transConv3[name.substr(11)];
        else
            throw std::runtime_error("No such layer in Generator: " + name);
    }
};


class Discriminator : public NeuralNet
{
    // Architecture: Input(28,28,1) ->
    //               Conv(64) -> LeakyReLU ->
    //               Conv(128) -> LeakyReLU ->
    //               Flatten -> FC(1) -> Sigmoid

    ConvolutionNode conv1;
    LeakyReluNode lrelu1;

    ConvolutionNode conv2;
    LeakyReluNode lrelu2;

    FlattenNode flatten;
    FullyConnectedNode fc;
    SigmoidNode sigmoid;

public:
    Discriminator(Device& device)
    : NeuralNet(device, 1, 1)
    , conv1(1, 64, 5)      // 31x31x1 -> 15x15x64 (changed from 4 to 5)
    , lrelu1(0.2f)
    , conv2(64, 128, 5)    // 15x15x64 -> 7x7x128 (changed from 4 to 5)
    , lrelu2(0.2f)
    , fc(7*7*128, 1)
    {
        // Connect the graph
        input(0) - conv1 - lrelu1 -
        conv2 - lrelu2 -
        flatten - fc - sigmoid - output(0);
    }

    // Access to weights (for loading/saving)
    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("conv1."))
            return conv1[name.substr(6)];
        else if (name.starts_with("conv2."))
            return conv2[name.substr(6)];
        else if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in Discriminator: " + name);
    }
};


class SimpleGAN
{
    Device& device;
    Generator generator;
    Discriminator discriminator;

    float learningRate;
    uint32_t batchSize;

public:
    SimpleGAN(Device& device, float lr = 0.0002f, uint32_t batch = 1)
    : device(device)
    , generator(device)
    , discriminator(device)
    , learningRate(lr)
    , batchSize(batch)
    {
    }

    // Generate fake images
    Tensor generate(uint32_t numImages = 1)
    {
        Tensor noise = generator.generateNoise(numImages);
        return generator(noise)[0];
    }

    // Discriminate real/fake (returns probability of being real)
    Tensor discriminate(const Tensor& images)
    {
        return discriminator(images)[0];
    }

    // Training step (to be implemented when backward pass is added)
    void trainStep(const std::vector<Tensor>& realImages)
    {
        // TODO: Implement training when backward pass is available
        // 1. Train discriminator on real images (label=1)
        // 2. Train discriminator on fake images (label=0)
        // 3. Train generator to fool discriminator (label=1 for fake images)
        throw std::runtime_error("Training not yet implemented - backward pass needed");
    }

    Generator& getGenerator() { return generator; }
    Discriminator& getDiscriminator() { return discriminator; }
};


// Large-scale Generator for performance benchmarking
// Deeper network with more channels for better performance comparison
class LargeGenerator : public NeuralNet
{
    // Architecture: Latent(256) -> FC(8*8*256) -> Reshape(8,8,256) ->
    //               TransConv(256->256) -> BN -> LeakyReLU ->
    //               TransConv(256->128) -> BN -> LeakyReLU ->
    //               TransConv(128->128) -> BN -> LeakyReLU ->
    //               TransConv(128->64) -> BN -> LeakyReLU ->
    //               TransConv(64->3) -> Tanh
    // Output: ~64x64x3

    FullyConnectedNode fc;
    ReshapeNode reshape;

    TransposeConvNode transConv1;
    BatchNormNode bn1;
    LeakyReluNode lrelu1;

    TransposeConvNode transConv2;
    BatchNormNode bn2;
    LeakyReluNode lrelu2;

    TransposeConvNode transConv3;
    BatchNormNode bn3;
    LeakyReluNode lrelu3;

    TransposeConvNode transConv4;
    BatchNormNode bn4;
    LeakyReluNode lrelu4;

    TransposeConvNode transConv5;
    TanhNode tanh;

public:
    LargeGenerator(Device& device)
    : NeuralNet(device, 1, 1)
    , fc(256, 8*8*256)
    , reshape({8, 8, 256})
    , transConv1(256, 256, 5, 2)  // 8x8 -> 17x17
    , bn1(256)
    , lrelu1(0.2f)
    , transConv2(256, 128, 5, 2)  // 17x17 -> 35x35
    , bn2(128)
    , lrelu2(0.2f)
    , transConv3(128, 128, 5, 2)  // 35x35 -> 71x71
    , bn3(128)
    , lrelu3(0.2f)
    , transConv4(128, 64, 3, 1)   // 71x71 -> 71x71
    , bn4(64)
    , lrelu4(0.2f)
    , transConv5(64, 3, 3, 1)     // 71x71 -> 71x71
    {
        // Connect the graph
        input(0) - fc - reshape -
        transConv1 - bn1 - lrelu1 -
        transConv2 - bn2 - lrelu2 -
        transConv3 - bn3 - lrelu3 -
        transConv4 - bn4 - lrelu4 -
        transConv5 - tanh - output(0);
    }

    Tensor generateNoise(uint32_t batchSize = 1)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> noise(batchSize * 256);
        for (auto& val : noise)
            val = dist(gen);

        return Tensor(batchSize, 256).set(noise);
    }
};


// Large-scale Discriminator for performance benchmarking
class LargeDiscriminator : public NeuralNet
{
    // Architecture: Input(71,71,3) ->
    //               Conv(3->64) -> LeakyReLU ->
    //               Conv(64->128) -> LeakyReLU ->
    //               Conv(128->128) -> LeakyReLU ->
    //               Conv(128->256) -> LeakyReLU ->
    //               Flatten -> FC(256) -> LeakyReLU -> FC(1) -> Sigmoid

    ConvolutionNode conv1;
    LeakyReluNode lrelu1;

    ConvolutionNode conv2;
    LeakyReluNode lrelu2;

    ConvolutionNode conv3;
    LeakyReluNode lrelu3;

    ConvolutionNode conv4;
    LeakyReluNode lrelu4;

    FlattenNode flatten;
    FullyConnectedNode fc1;
    LeakyReluNode lrelu5;
    FullyConnectedNode fc2;
    SigmoidNode sigmoid;

public:
    LargeDiscriminator(Device& device)
    : NeuralNet(device, 1, 1)
    , conv1(3, 64, 5)       // 71x71 -> 35x35
    , lrelu1(0.2f)
    , conv2(64, 128, 5)     // 35x35 -> 17x17
    , lrelu2(0.2f)
    , conv3(128, 128, 5)    // 17x17 -> 8x8
    , lrelu3(0.2f)
    , conv4(128, 256, 5)    // 8x8 -> 4x4
    , lrelu4(0.2f)
    , fc1(4*4*256, 256)
    , lrelu5(0.2f)
    , fc2(256, 1)
    {
        // Connect the graph
        input(0) - conv1 - lrelu1 -
        conv2 - lrelu2 -
        conv3 - lrelu3 -
        conv4 - lrelu4 -
        flatten - fc1 - lrelu5 - fc2 - sigmoid - output(0);
    }
};


#endif // GAN_H
