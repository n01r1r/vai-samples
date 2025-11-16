#ifndef MOBILENET_VULKAN_APP_H
#define MOBILENET_VULKAN_APP_H

#include <vulkan/vulkan_core.h>

namespace mobilenet {

class VulkanApp {
public:
    VulkanApp();
    ~VulkanApp();
    
    bool initialize();
    void cleanup();
    
private:
    VkInstance instance_;
    bool initialized_;
};

} // namespace mobilenet

#endif // MOBILENET_VULKAN_APP_H

