#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>



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


Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json) // srcImage layout: [H][W][C]
{
    NeuralNet mnistNet(gDevice, 1, 1); 

    auto conv1 = ConvolutionNode(1, 32, 3); 
    auto relu1 = ReluNode();
    auto maxpool1 = MaxPoolingNode(2);
    auto conv2 = ConvolutionNode(32, 64, 3);
    auto relu2 = ReluNode();
    auto maxpool2 = MaxPoolingNode(2);
    auto flatten = FlattenNode();
    auto fc = FullyConnectedNode(7*7*64, 10); 

    mnistNet.input(0) - conv1 - relu1 - maxpool1 - conv2 - relu2 - maxpool2 - flatten - fc - mnistNet.output(0);

    // conv1["weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);                     // 32 x 1 x 3 x 3 => 1*3*3 x 32
    // conv1["bias"] = Tensor(json["layer1.0.bias"]);                                                          // 32
    // conv2["weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);                    // 64 x 32 x 3 x 3 => 32*3*3 x 64
    // conv2["bias"] = Tensor(json["layer2.0.bias"]);                                                          // 64                                        
    // fc["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);     // 10 x 64*7*7 => 7*7*64 x 10
    // fc["bias"] = Tensor(json["fc.bias"]);                                                                   // 10
    //
    // return mnistNet(Tensor(28, 28, 1).set(srcImage))[0];

    Tensor conv1_weight = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);                     // 32 x 1 x 3 x 3 => 1*3*3 x 32
    Tensor conv1_bias = Tensor(json["layer1.0.bias"]);                                                          // 32
    Tensor conv2_weight = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);                    // 64 x 32 x 3 x 3 => 32*3*3 x 64
    Tensor conv2_bias = Tensor(json["layer2.0.bias"]);                                                          // 64
    Tensor fc_weight = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);     // 10 x 64*7*7 => 7*7*64 x 10
    Tensor fc_bias = Tensor(json["fc.bias"]);                                                                   // 10

    uint32_t iter = 1000;
    Tensor result;

    {
        TimeChecker timer("MNIST evaluation: {} iterations", iter);
        for (uint32_t i = 0; i < iter; ++i) 
        {
            conv1["weight"] = conv1_weight;
            conv1["bias"] = conv1_bias;
            conv2["weight"] = conv2_weight;
            conv2["bias"] = conv2_bias;
            fc["weight"] = fc_weight;
            fc["bias"] = fc_bias;
            result = mnistNet(Tensor(28, 28, 1).set(srcImage))[0];
        }
    }

    return result;
}


void test()
{
    const uint32_t channels = 1;
    auto [srcImage, width, height] = readImage<channels>(PROJECT_ROOT_DIR"/data/0.png");
    _ASSERT(width == 28 && height == 28);
    _ASSERT(width * height * channels == srcImage.size());

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");
    Tensor eval = eval_mnist(inputData, json);

    vk::Buffer outBuffer = gDevice.createBuffer({
        10 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    vk::Buffer evalBuffer = eval.buffer();
    gDevice.newCommandBuffer(queue_compute)
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




















































// void test()
// {
//     const uint32_t channels = 1;
//     auto [srcImage, width, height] = readImage<channels>(PROJECT_ROOT_DIR"/data/9.png");
//     _ASSERT(width == MNIST_IMAGE_WIDTH && height == MNIST_IMAGE_HEIGHT);
//     _ASSERT(width * height * channels != srcImage.size());

//     std::vector<float> inputData = std::ranges::to<std::vector<float>>(
//         srcImage | std::views::transform([](uint8_t v) { return v / 255.0f; }) );

//     Device device = VulkanApp::get().createDevice({.supportPresent = false});
//     auto dstPool = device.createDescriptorPool({
//         .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 10}, 
//         .maxSets = 8
//     });
//     auto cmdPool = device.createCommandPool(queue_compute);    

//     constexpr uint32_t H = MNIST_IMAGE_HEIGHT;
//     constexpr uint32_t W = MNIST_IMAGE_WIDTH;
//     constexpr uint32_t C = MNIST_IMAGE_CHANNELS;
//     constexpr uint32_t K = MNIST_KERNEL_WIDTH;
//     constexpr uint32_t HW = H * W;
//     constexpr uint32_t C2 = MNIST_KERNEL_CHANNELS;
//     constexpr uint32_t CKK = C * K * K;
//     constexpr uint32_t H2 = H / poolSize;
//     constexpr uint32_t W2 = W / poolSize;
//     constexpr uint32_t H2W2 = H2 * W2;
//     constexpr uint32_t C2KK = C2 * K * K;

//     Buffer imData = device.createBuffer({
//         HW * C * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });
//     memcpy(imData.map(), inputData.data(), inputData.size() * sizeof(float));

//     Buffer colData = device.createBuffer({
//         HW * CKK * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     Buffer weightData = device.createBuffer({
//         CKK * C2 * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     Buffer biasData = device.createBuffer({
//         C2 * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     Buffer convOutData = device.createBuffer({
//         HW * C2 * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     Buffer reluOutData = device.createBuffer({
//         HW * C2 * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     Buffer maxpoolOutData = device.createBuffer({
//         H2 * W2 * C2 * sizeof(float), 
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
//         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//     });

//     /* 
//     im2col:
//     -  input memory layout: [H][W][C]
//     - output memory layout: [H][W][C][K*K]
//     - (H*W)x(C*K*K) invocations in a dispatch
//     - K must be odd
//     */
//     ComputePipeline im2col = device.createComputePipeline({ im2col_srcCode });
//     auto im2col_dstSet = im2col.descSetLayout(0).newDescSet(dstPool).write({colData, imData});
//     uint32_t im2col_constants[] = { H, W, C, K };

//     /*
//     gemm:
//     -  input memory layout: [HW][CKK]  
//     - output memory layout: [HW][C2]  
//     - weight memory layout: [CKK][C2]  
//     - bias: [C2]
//     */
//     ComputePipeline gemm = device.createComputePipeline({ gemm_srcCode });
//     auto gemm_dstSet = gemm.descSetLayout(0).newDescSet(dstPool).write({convOutData, colData, weightData, biasData});
//     uint32_t gemm_constants[] = { HW, C2, CKK };

//     /*
//     relu:
//     */
//     ComputePipeline relu = device.createComputePipeline({ relu_srcCode });
//     auto relu_dstSet = relu.descSetLayout(0).newDescSet(dstPool).write({reluOutData, convOutData});
//     uint32_t relu_constants[] = { HW * C2 };

//     /*
//     maxpool:
//     -  input memory layout: [H][W][C2]
//     - output memory layout: [H2][W2][C2]
//     */
//     ComputePipeline maxpool = device.createComputePipeline({ maxpool_srcCode });
//     auto maxpool_dstSet = maxpool.descSetLayout(0).newDescSet(dstPool).write({maxpoolOutData, reluOutData});
//     uint32_t maxpool_constants[] = { H, W, C2, poolSize };


//     /* im2col: [HW][C] -> [HW][CKK] */
//     auto cmdBuffer1 = cmdPool.newCommandBuffer()
//         .begin()
//             .bindPipeline(im2col)
//             .bindDescSets({im2col_dstSet})
//             .setPushConstants(0, sizeof(im2col_constants), &im2col_constants)
//             .dispatch(HW, CKK)
//         .end();

//     /* gemm: [HW][CKK] -> [HW][C2] */
//     auto cmdBuffer2 = cmdPool.newCommandBuffer()
//         .begin()
//             .bindPipeline(gemm)
//             .bindDescSets({gemm_dstSet})
//             .setPushConstants(0, sizeof(gemm_constants), &gemm_constants)
//             .dispatch(HW, C2)
//         .end();

//     /* relu: [HW*C2] -> [HW*C2] */
//     auto cmdBuffer3 = cmdPool.newCommandBuffer()
//         .begin()
//             .bindPipeline(relu)
//             .bindDescSets({relu_dstSet})
//             .setPushConstants(0, sizeof(relu_constants), &relu_constants)
//             .dispatch(HW * C2)
//         .end();

//     /* maxpool: [H][W][C2] -> [H2][W2][C2] */
//     auto cmdBuffer4 = cmdPool.newCommandBuffer()
//         .begin()
//             .bindPipeline(maxpool)
//             .bindDescSets({maxpool_dstSet})
//             .setPushConstants(0, sizeof(maxpool_constants), &maxpool_constants)
//             .dispatch(H / poolSize, W / poolSize, C2)
//         .end();

//     device.queue() << (cmdBuffer1, cmdBuffer2, cmdBuffer3, cmdBuffer4) << waiting;

    


//     uint32_t im2col_constants2[] = { H2, W2, C2, K };
    
//     /* im2col: [H2W2][C2] -> [H2W2][C2KK] */
//     auto cmdBuffer5 = cmdPool.newCommandBuffer()
//         .begin()
//             .bindPipeline(im2col)
//             .bindDescSets({im2col_dstSet})
//             .setPushConstants(0, sizeof(im2col_constants2), &im2col_constants2)
//             .dispatch(H2W2, C2KK)
//         .end();

// }