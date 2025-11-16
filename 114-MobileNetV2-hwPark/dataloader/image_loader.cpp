#include "dataloader/image_loader.h"

namespace mobilenet {

ImageLoader::ImageLoader() : targetWidth_(224), targetHeight_(224) {
}

Tensor ImageLoader::loadImage(const std::string& imagePath) {
    // TODO: Implement image loading
    return Tensor({targetHeight_, targetWidth_, 3});
}

Tensor ImageLoader::preprocess(const Tensor& image) {
    // TODO: Implement preprocessing (normalize, etc.)
    return image;
}

} // namespace mobilenet

