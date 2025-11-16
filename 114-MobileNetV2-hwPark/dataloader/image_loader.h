#ifndef MOBILENET_IMAGE_LOADER_H
#define MOBILENET_IMAGE_LOADER_H

#include "core/tensor.h"
#include <string>

namespace mobilenet {

class ImageLoader {
public:
    ImageLoader();
    ~ImageLoader() = default;
    
    Tensor loadImage(const std::string& imagePath);
    Tensor preprocess(const Tensor& image);
    
private:
    uint32_t targetWidth_;
    uint32_t targetHeight_;
};

} // namespace mobilenet

#endif // MOBILENET_IMAGE_LOADER_H

