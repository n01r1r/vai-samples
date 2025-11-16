#include "core/neural_net.h"
#include <memory>

namespace mobilenet {

Node::Node(const std::string& name) : name_(name) {
}

NeuralNet::NeuralNet() {
}

void NeuralNet::addNode(std::unique_ptr<Node> node) {
    nodes_.push_back(std::move(node));
}

void NeuralNet::forward() {
    // TODO: Implement forward pass
}

} // namespace mobilenet

