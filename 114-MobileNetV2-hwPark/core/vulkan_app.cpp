#include "core/vulkan_app.h"
#include "core/error.h"

namespace mobilenet {

VulkanApp::VulkanApp() : instance_(VK_NULL_HANDLE), initialized_(false) {
}

VulkanApp::~VulkanApp() {
    cleanup();
}

bool VulkanApp::initialize() {
    if (initialized_) {
        return true;
    }
    
    // TODO: Initialize Vulkan instance and device
    initialized_ = true;
    return true;
}

void VulkanApp::cleanup() {
    if (instance_ != VK_NULL_HANDLE) {
        // TODO: Destroy Vulkan instance
        // vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    initialized_ = false;
}

} // namespace mobilenet

