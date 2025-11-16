#ifndef MOBILENET_NEURAL_NET_H
#define MOBILENET_NEURAL_NET_H

#include "core/tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace mobilenet {

class Node {
public:
    Node(const std::string& name);
    virtual ~Node() = default;
    
    const std::string& name() const { return name_; }
    
private:
    std::string name_;
};

class NeuralNet {
public:
    NeuralNet();
    ~NeuralNet() = default;
    
    void addNode(std::unique_ptr<Node> node);
    void forward();
    
private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

} // namespace mobilenet

#endif // MOBILENET_NEURAL_NET_H

