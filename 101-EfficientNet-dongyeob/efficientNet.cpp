#include "efficientNet.h"


EfficientNet::EfficientNet(Device& device, const std::vector<MBConvConfig>& blockConfigs, uint32_t numClasses)
: NeuralNet(device, 1, 1)
, stem(3, 32, 3)
, globalAvgPool()
, classifier(1280, numClasses)
{
    // TODO: Initialize MBConv blocks from configurations
    for (const auto& config : blockConfigs)
    {
        mbconvBlocks.push_back(std::make_unique<MBConvBlockNode>(config));
    }
    
    // TODO: Connect all nodes
}

Tensor& EfficientNet::operator[](const std::string& name)
{
    // TODO: Route weight access to appropriate layers
    
    if (name.starts_with("stem."))
    {
        return stem[name.substr(5)];
    }
    else if (name.starts_with("blocks."))
    {
        // TODO: Parse block index and route to appropriate MBConv block
        throw std::runtime_error("Block weight access not yet implemented: " + name);
    }
    else if (name.starts_with("classifier."))
    {
        return classifier[name.substr(11)];
    }
    else
    {
        throw std::runtime_error("Unknown weight name in EfficientNet: " + name);
    }
}

