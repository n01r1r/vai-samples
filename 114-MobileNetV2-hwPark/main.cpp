#include <iostream>
#include "core/error.h"
#include "core/vulkan_app.h"
#include "core/tensor.h"
#include "core/neural_net.h"
#include "nodes/base_node.h"
#include "model/mobilenet_v2.h"
#include "dataloader/image_loader.h"
#include "utils/json_parser.h"

int main()
{
    std::cout << "Hello, MobileNetV2-hwPark" << std::endl;
    
    mobilenet::VulkanApp app;
    mobilenet::Tensor tensor({1, 3, 224, 224});
    mobilenet::NeuralNet net;
    mobilenet::BaseNode node("test");
    mobilenet::MobileNetV2 model;
    mobilenet::ImageLoader loader;
    mobilenet::JsonParser parser;
    
    std::cout << "All modules loaded successfully!" << std::endl;
    return 0;
}