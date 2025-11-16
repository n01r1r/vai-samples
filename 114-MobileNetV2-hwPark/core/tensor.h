#ifndef MOBILENET_TENSOR_H
#define MOBILENET_TENSOR_H

#include <vector>
#include <cstdint>

namespace mobilenet {

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<uint32_t>& shape);
    ~Tensor() = default;
    
    const std::vector<uint32_t>& shape() const { return shape_; }
    size_t numElements() const;
    
private:
    std::vector<uint32_t> shape_;
    std::vector<float> data_;
};

} // namespace mobilenet

#endif // MOBILENET_TENSOR_H

