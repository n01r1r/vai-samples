#include "core/tensor.h"

namespace mobilenet {

Tensor::Tensor() : shape_({}) {
}

Tensor::Tensor(const std::vector<uint32_t>& shape) : shape_(shape) {
    size_t total = 1;
    for (uint32_t dim : shape) {
        total *= dim;
    }
    data_.resize(total, 0.0f);
}

size_t Tensor::numElements() const {
    size_t total = 1;
    for (uint32_t dim : shape_) {
        total *= dim;
    }
    return total;
}

} // namespace mobilenet

