#ifndef MOBILENET_BASE_NODE_H
#define MOBILENET_BASE_NODE_H

#include "core/neural_net.h"

namespace mobilenet {

class BaseNode : public Node {
public:
    BaseNode(const std::string& name);
    virtual ~BaseNode() = default;
    
    virtual void prepare() {}
    virtual void forward() {}
};

} // namespace mobilenet

#endif // MOBILENET_BASE_NODE_H

