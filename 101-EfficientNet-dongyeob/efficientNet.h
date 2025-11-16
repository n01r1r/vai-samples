#ifndef EFFICIENTNET_H
#define EFFICIENTNET_H

#include "neuralNet.h"
#include "neuralNodes.h"
#include <vector>
#include <memory>


class EfficientNet : public NeuralNet
{
    // Stem layer
    ConvBNSwishNode stem;
    
    // MBConv blocks
    std::vector<std::unique_ptr<MBConvBlockNode>> mbconvBlocks;
    
    // Head layers
    GlobalAvgPoolNode globalAvgPool;
    FullyConnectedNode classifier;

public:
    EfficientNet(Device& device, const std::vector<MBConvConfig>& blockConfigs, uint32_t numClasses = 1000);
    
    Tensor& operator[](const std::string& name);
};

#endif // EFFICIENTNET_H

