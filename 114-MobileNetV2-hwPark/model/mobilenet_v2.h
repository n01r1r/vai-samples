#ifndef MOBILENET_MOBILENET_V2_H
#define MOBILENET_MOBILENET_V2_H

#include "core/neural_net.h"

namespace mobilenet {

class MobileNetV2 : public NeuralNet {
public:
    MobileNetV2();
    ~MobileNetV2() = default;
    
    void initialize();
    void loadWeights(const std::string& weightsPath);
};

} // namespace mobilenet

#endif // MOBILENET_MOBILENET_V2_H

