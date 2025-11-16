#include <GLFW/glfw3.h>
#include <array>
#include <deque>
#include <map>
#include <set>
#include <ranges>   // std::views::filter
#include <algorithm>// std::all_of, std::any_of
#include <fstream>
#include <cstring>  // strcmp, memcpy
#include "error.h"
#include "templateHelper.h"
#include "vulkanApp.h"


std::vector<uint32_t> glsl2spv(VkShaderStageFlags stage, const char* shaderSource);
void* createReflectShaderModule(const std::vector<uint32_t>& spvBinary);
void destroyReflectShaderModule(void* pModule);
vk::PipelineLayoutDesc extractPipelineLayoutDesc(const void* pModule);
std::array<uint32_t, 3> extractWorkGroupSize(const void* pModule);

using namespace vk;


std::string operator""_file2str(const char* filename, size_t)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    ASSERT_(file.is_open());

    size_t fileSize = (size_t)file.tellg();

    std::string str;str.resize(fileSize);
    file.seekg(0, std::ios::beg);
    file.read(str.data(), fileSize);
    return str;
}

static void printQueueFamily(uint32_t qfIndex, uint32_t qCount, VkQueueFlags qFlags)
{
    printf("[Queue Family %u] Queue Count: %u, Queue Flags: %s%s%s%s%s\n", qfIndex, qCount,
        (qFlags & VK_QUEUE_GRAPHICS_BIT) ? "Graphics " : "",
        (qFlags & VK_QUEUE_COMPUTE_BIT) ? "Compute " : "",
        (qFlags & VK_QUEUE_TRANSFER_BIT) ? "Transfer " : "",
        (qFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "SparseBinding " : "",
        (qFlags & VK_QUEUE_PROTECTED_BIT) ? "Protected " : "");
}

static VkInstance createVkInstance()
{
    static bool first = true;
    ASSERT_(first);
    first = false;

    uint32_t extensionCount = 0;
    const char** extensionNames = glfwGetRequiredInstanceExtensions(&extensionCount);
    std::vector<const char*> extensions(extensionNames, extensionNames + extensionCount);

    std::vector<const char*> requiredLayers;
#ifndef NDEBUG
    requiredLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    auto allLayers = arrayFrom(vkEnumerateInstanceLayerProperties);
    bool ok = std::all_of(requiredLayers.begin(), requiredLayers.end(), [&](const char* layer) {
        return std::any_of(allLayers.begin(), allLayers.end(), [&](const VkLayerProperties& porps) {
            return strcmp(layer, porps.layerName) == 0;
        });
    });
    ASSERT_(ok);
    
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan App",
        .apiVersion = VK_API_VERSION_1_3
    };

    return create<VkInstance>({
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = (uint32_t)requiredLayers.size(),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = (uint32_t)extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    });
}

static void printGpuInfo(uint32_t order, VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    printf("[GPU %d] Device Name: %-30s API Version: %d.%d.%d  Driver Version: %d.%d.%d  Device Type: %-15s\n",
        order,
        props.deviceName,
        VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
        VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion),
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "Integrated GPU" :
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" :
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "Virtual GPU" :
                    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU" : "Other");
}

static bool deviceSupportsExtensions(
    VkPhysicalDevice physicalDevice, 
    const std::vector<const char*>& extensions)
{
    auto deviceExtensions = arrayFrom(vkEnumerateDeviceExtensionProperties, physicalDevice, nullptr);

    return std::all_of(extensions.begin(), extensions.end(), [&](const char* reqExtension) {
        return std::any_of(deviceExtensions.begin(), deviceExtensions.end(), [&](const VkExtensionProperties& props) {
            return strcmp(props.extensionName, reqExtension) == 0;
        });
    });
}

static inline VkPhysicalDeviceMemoryProperties getMemorySpec(VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceMemoryProperties spec;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &spec);
    return spec;
}

template <typename VkResource>
static std::pair<VkMemoryAllocateInfo, VkMemoryPropertyFlags> getMemoryAllocInfo(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkResource resource,
    VkMemoryPropertyFlags reqMemProps)
{
    static VkPhysicalDevice cached = VK_NULL_HANDLE;
    static VkPhysicalDeviceMemoryProperties spec;
    if (cached != physicalDevice) {
        spec = getMemorySpec(physicalDevice);
        cached = physicalDevice;
    }

    VkMemoryRequirements memRequirements;
    if constexpr (std::is_same_v<VkResource, VkBuffer>) 
        vkGetBufferMemoryRequirements(device, resource, &memRequirements);  // It should be removed for performance!
    else if constexpr (std::is_same_v<VkResource, VkImage>)
        vkGetImageMemoryRequirements(device, resource, &memRequirements);
    else 
        static_assert(false, "Invalid VkResource type");

    /*
    In Vulkan specification, the memoryTypes array is ordered by the following rules:
        For each pair of elements X and Y returned in memoryTypes, X must be placed at a lower index
        position than Y if:
            • the set of bit flags returned in the propertyFlags member of X is a strict subset of the set of bit
            flags returned in the propertyFlags member of Y; or
    */
    uint32_t i = 0;
    for ( ; i < spec.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1<<i)) != 0
            && (spec.memoryTypes[i].propertyFlags & reqMemProps) == reqMemProps) 
            break;
    }
    ASSERT_(i != spec.memoryTypeCount);

    return { 
        {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = i,
        }, 
        spec.memoryTypes[i].propertyFlags
    };
}

template <typename T>
using Groups = std::map<std::string, std::vector<T>>;

#define GROUP_PERMANENT "permanent"


/////////////////////////////////////////////////////////////////////////////////////////
// Impl classes
/////////////////////////////////////////////////////////////////////////////////////////
struct VulkanApp::Impl {
    const VkInstance instance;
    std::vector<VkPhysicalDevice> physicalDevices;
    std::map<VkPhysicalDevice, std::vector<VkQueueFamilyProperties>> qfProps;
    std::map<VkPhysicalDevice, std::vector<bool>> qfSupportPresent;
    std::map<VkPhysicalDevice, Device> devices;
    uint32_t defaultPhysicalDeviceIndex = 0;

    Impl(VkInstance instance) : instance(instance) {}
};


struct Device::Impl {
    const uint32_t qfIndex[queue_max];    // uint32_t(-1) if and only if DeviceSettings does not require the queue type
    const uint32_t qCount[queue_max];
    const std::vector<std::vector<Queue>> queues;
    const VkPhysicalDevice vkPhysicalDevice;
    const VkDevice vkDevice;
    // const VkInstance vkInstance;
    const VulkanApp& parent;

    Groups<DescriptorSetLayout> descSetLayouts = { {GROUP_PERMANENT, {} } };
    Groups<PipelineLayout> pipelineLayouts = { {GROUP_PERMANENT, {} } };
    // Groups<ShaderModule> shaderModules;
    Groups<ComputePipeline> computePipelines = { {GROUP_PERMANENT, {} } };
    Groups<DescriptorPool> descPools = { {GROUP_PERMANENT, {} } };
    Groups<CommandPool> cmdPools = { {GROUP_PERMANENT, {} } };
    Groups<Fence> fences = { {GROUP_PERMANENT, {} } };
    Groups<Semaphore> semPools = { {GROUP_PERMANENT, {} } };
    Groups<Buffer> buffers = { {GROUP_PERMANENT, {} } };
    //Groups<Image> images;

    CommandPool defaultCmdPool[queue_max][8] = {};

    Impl(VkPhysicalDevice vkPhysicalDevice,   
        VkDevice vkDevice, 
        const VulkanApp& parent,
        uint32_t graphicsQfIndex,
        uint32_t computeQfIndex,
        uint32_t transferQfIndex,
        std::vector<std::vector<Queue>>&& queues) 
    : vkPhysicalDevice(vkPhysicalDevice)
    , vkDevice(vkDevice)
    , parent(parent)
    , qfIndex{
        graphicsQfIndex, 
        computeQfIndex, 
        transferQfIndex}
        , qCount{
            graphicsQfIndex != uint32_t(-1) ? (uint32_t) queues[graphicsQfIndex].size() : 0,
            computeQfIndex != uint32_t(-1) ? (uint32_t) queues[computeQfIndex].size() : 0,
            transferQfIndex != uint32_t(-1) ? (uint32_t) queues[transferQfIndex].size() : 0}
    , queues(std::move(queues))
    {}
    ~Impl();                               
};


struct Queue::Impl {
    const VkQueue vkQueue;
    const uint32_t qfIndex;
    const uint32_t index;
    const float priority;

    Impl(
        VkQueue vkQueue,  
        uint32_t qfIndex, 
        uint32_t index,
        float priority)
    : vkQueue(vkQueue)
    , qfIndex(qfIndex)
    , index(index)
    , priority(priority) {}
};


struct CommandPool::Impl {
    const VkDevice vkDevice;
    const Device device;
    const VkCommandPool vkCmdPool;
    const VkCommandPoolCreateFlags flags;
    const uint32_t qfIndex;
    const QueueType type;
    std::deque<CommandBuffer> cmdBuffers;

    Impl(VkDevice vkDevice,
        Device device,
        VkCommandPool vkCmdPool, 
        VkCommandPoolCreateFlags flags,
        uint32_t qfIndex,
        QueueType type)
    : vkDevice(vkDevice)
    , device(device)
    , vkCmdPool(vkCmdPool)
    , flags(flags)
    , qfIndex(qfIndex)
    , type(type) {} 
    ~Impl();
};


struct CommandBuffer::Impl {
    const VkCommandBuffer vkCmdBuffer;
    const Device device;
    const uint32_t qfIndex;
    const QueueType type;
    Pipeline boundPipeline;
    Queue lastSubmittedQueue;

    Impl( 
        VkCommandBuffer vkCmdBuffer, 
        Device device,
        uint32_t qfIndex,
        QueueType type)
    : vkCmdBuffer(vkCmdBuffer)
    , device(device)
    , qfIndex(qfIndex)
    , type(type)
    {}
};


CommandPool::Impl::~Impl()
{
    for (auto& cmdBuffer : cmdBuffers) 
            delete cmdBuffer.impl;
    vkDestroyCommandPool(vkDevice, vkCmdPool, nullptr); 
}


struct Fence::Impl {
    const VkDevice vkDevice;
    const VkFence vkFence;

    Impl(VkDevice vkDevice, 
        VkFence vkFence) 
    : vkDevice(vkDevice)
    , vkFence(vkFence) {}
    ~Impl() {
        vkDestroyFence(vkDevice, vkFence, nullptr);
    }
};


struct Semaphore::Impl {
    const VkDevice vkDevice;
    const VkSemaphore vkSemaphore;

    Impl(VkDevice vkDevice, 
        VkSemaphore vkSemaphore) 
    : vkDevice(vkDevice)
    , vkSemaphore(vkSemaphore) {}
    ~Impl() {
        vkDestroySemaphore(vkDevice, vkSemaphore, nullptr);
    }
};


struct ComputePipeline::Impl {
    const VkDevice vkDevice;
    const VkPipeline vkPipeline;
    const PipelineLayout layout;
    const std::array<uint32_t, 3> workGroupSize;

    Impl(VkDevice vkDevice,             
        VkPipeline vkPipeline, 
        PipelineLayout layout,
        uint32_t sizeX,
        uint32_t sizeY,
        uint32_t sizeZ) 
    : vkDevice(vkDevice)
    , vkPipeline(vkPipeline)
    , layout(layout)
    , workGroupSize{sizeX, sizeY, sizeZ} {}
    ~Impl() {                           
        vkDestroyPipeline(vkDevice, vkPipeline, nullptr); 
    }
};


struct Buffer::Impl {
    const VkDevice vkDevice;
    const VkBuffer vkBuffer;
    const VkDeviceMemory vkMemory;
    const uint64_t size;
    const VkBufferUsageFlags usage;
    const VkMemoryPropertyFlags reqMemProps;
    const VkMemoryPropertyFlags memProps;
    uint8_t* mapped = nullptr;
    uint64_t mappedOffset = 0;  // used for debug
    uint64_t mappedSize = 0;    // used for debug

    Impl(VkDevice vkDevice, 
        VkBuffer vkBuffer, 
        VkDeviceMemory vkMemory, 
        uint64_t size, 
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags reqMemProps,
        VkMemoryPropertyFlags memProps)
    : vkDevice(vkDevice)
    , vkBuffer(vkBuffer)
    , vkMemory(vkMemory)
    , size(size)
    , usage(usage)
    , reqMemProps(reqMemProps)
    , memProps(memProps) {}
    ~Impl() {                           
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr); 
        vkFreeMemory(vkDevice, vkMemory, nullptr); 
    }

    VkMappedMemoryRange getRange(uint64_t offset, uint64_t size);

};


struct DescriptorSetLayout::Impl {
    const VkDevice vkDevice;
    const VkDescriptorSetLayout vkSetLayout;
    const std::map<uint32_t, VkDescriptorSetLayoutBinding> bindingInfos;

    Impl(VkDevice vkDevice,                    
        VkDescriptorSetLayout vkSetLayout,
        std::vector<VkDescriptorSetLayoutBinding> setInfo) 
    : vkDevice(vkDevice)
    , vkSetLayout(vkSetLayout)
    , bindingInfos([&setInfo]() {
        std::map<uint32_t, VkDescriptorSetLayoutBinding> map;
        for (auto& bindingInfo : setInfo) 
            map[bindingInfo.binding] = bindingInfo;
        return map; }())
    {}
    ~Impl() {                                  
        vkDestroyDescriptorSetLayout(vkDevice, vkSetLayout, nullptr); 
    }
};


struct PipelineLayout::Impl {
    const VkDevice vkDevice;
    const VkPipelineLayout vkPipeLayout;
    const std::vector<DescriptorSetLayout> setLayouts;
    const std::vector<PushConstantRange> pushConstants;

    Impl(VkDevice vkDevice,           
        VkPipelineLayout vkPipeLayout, 
        std::vector<DescriptorSetLayout> setLayouts,
        std::vector<PushConstantRange> pushConstants)
    : vkDevice(vkDevice)
    , vkPipeLayout(vkPipeLayout)
    , setLayouts(std::move(setLayouts))
    , pushConstants(std::move(pushConstants)) {}
    ~Impl() {                               
        vkDestroyPipelineLayout(vkDevice, vkPipeLayout, nullptr); 
    }
};


struct DescriptorPool::Impl {
    const VkDevice vkDevice;
    const VkDescriptorPool vkDescPool;
    std::deque<DescriptorSet> descSets;

    Impl(VkDevice vkDevice,             
        VkDescriptorPool vkDescPool) 
    : vkDevice(vkDevice)
    , vkDescPool(vkDescPool) {}
    ~Impl();
};


struct DescriptorSet::Impl {
    const VkDevice vkDevice;
    const VkDescriptorSet vkDescSet;
    const DescriptorSetLayout layout;

    Impl(VkDevice vkDevice,            
        VkDescriptorSet vkDescSet,
        DescriptorSetLayout layout) 
    : vkDevice(vkDevice)
    , vkDescSet(vkDescSet)
    , layout(layout) {}
};


DescriptorPool::Impl::~Impl()
{
    for (auto& descSet : descSets) 
        delete descSet.impl;
    vkDestroyDescriptorPool(vkDevice, vkDescPool, nullptr); 
}


/////////////////////////////////////////////////////////////////////////////////////////
// VulkanApp
/////////////////////////////////////////////////////////////////////////////////////////
VulkanApp::VulkanApp(DeviceSettings defaultSettings)
: impl(new Impl(createVkInstance())) 
{
    impl->physicalDevices = arrayFrom(vkEnumeratePhysicalDevices, impl->instance);
    _ASSERT(impl->physicalDevices.size() > 0);
    
    for (auto pd : impl->physicalDevices)
    {
        impl->qfProps[pd] = arrayFrom(vkGetPhysicalDeviceQueueFamilyProperties, pd);

        impl->qfSupportPresent[pd].resize(impl->qfProps[pd].size());
        for (uint32_t i = 0; i < impl->qfProps[pd].size(); ++i) 
        {
            impl->qfSupportPresent[pd][i] = glfwGetPhysicalDevicePresentationSupport(impl->instance, pd, i);
        }
    }

    if (impl->physicalDevices.size() == 1)
    {
        bool res = initDevice(0, defaultSettings);
        _ASSERT(res);
        impl->defaultPhysicalDeviceIndex = 0;
    }
    else
    {
        while(true)
        {
            int selected = -1;
            printf("Select a physical device (0-%d): ", (uint32_t)impl->physicalDevices.size() - 1);
            fflush(stdout);
            scanf("%d", &selected);
            if (selected < 0 || selected >= (int)impl->physicalDevices.size())
            {            
                fprintf(stderr, "[Error] Invalid index %d.\n", selected);
                continue;
            }
    
            if (initDevice((uint32_t)selected, defaultSettings))
            {
                impl->defaultPhysicalDeviceIndex = (uint32_t)selected;
                printf("[Info] Successfully initialized context for physical device %d.\n", selected);
                fflush(stdout);
                break;
            }

            fprintf(stderr, "[Error] Failed to initialize context for physical device %d. Please select again.\n", selected);
        }
    }
}

VulkanApp::~VulkanApp()
{
    for (auto& [pd, device] : impl->devices) 
        delete device.impl;
    vkDestroyInstance(impl->instance, nullptr); 
    delete impl;
}

VulkanApp& VulkanApp::get()
{
    static VulkanApp singleton;
    return singleton;
}

Device VulkanApp::device(uint32_t gpuIndex)
{
    ASSERT_(gpuIndex == uint32_t(-1) || gpuIndex < impl->physicalDevices.size());
    uint32_t idx = gpuIndex == uint32_t(-1) ? impl->defaultPhysicalDeviceIndex : gpuIndex;
    return impl->devices.at(impl->physicalDevices[idx]);
}

bool VulkanApp::initDevice(uint32_t gpuIndex, DeviceSettings settings)
{
    _ASSERT(gpuIndex < impl->physicalDevices.size());
    VkPhysicalDevice physicalDevice = impl->physicalDevices[gpuIndex];

    const std::vector<VkQueueFamilyProperties>& qfProps = impl->qfProps[physicalDevice];
    const std::vector<bool>& qfSupportPresent = impl->qfSupportPresent[physicalDevice];
    _ASSERT(qfProps.size() > 0);
    _ASSERT(qfSupportPresent.size() == qfProps.size());

    std::vector<const char*> reqExtentions;
    if (settings.supportPresent) 
    {
        bool presentSupport = std::any_of(
            qfSupportPresent.begin(), 
            qfSupportPresent.end(), 
            [](bool support) { return support; }
        );

        if (!presentSupport) 
        {
            fprintf(stderr, "[Error] The selected physical device does not support presentation.\n");
            return false;
        }

        reqExtentions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    reqExtentions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    if (!deviceSupportsExtensions(physicalDevice, reqExtentions))
    {
        fprintf(stderr, "[Error] The selected physical device does not support the required extensions.\n");
        return false;
    }
    
    std::vector<uint32_t> qfIndices[queue_max];
    for (uint32_t i = 0; i < qfProps.size(); i++) 
    {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT 
            && qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) 
            qfIndices[queue_graphics].push_back(i);
        
        else if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) 
            qfIndices[queue_compute].push_back(i);
        
        else if (qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) 
            qfIndices[queue_transfer].push_back(i);
    }

    if(qfIndices[queue_graphics].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Graphics (and Compute) operations were detected:\n");
        for (uint32_t i : qfIndices[queue_graphics])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }

    if(qfIndices[queue_compute].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Compute operations were detected:\n");
        for (uint32_t i : qfIndices[queue_compute])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }

    if(qfIndices[queue_transfer].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Transfer operations were detected:\n");
        for (uint32_t i : qfIndices[queue_transfer])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }

    uint32_t qfIndex[queue_max] = { uint32_t(-1), uint32_t(-1), uint32_t(-1) };

    if (settings.requireGrapicsQueues) 
    {
        if (!qfIndices[queue_graphics].empty()) 
        {
            qfIndex[queue_graphics] = qfIndices[queue_graphics][0];
        } 
        else 
        {
            fprintf(stderr, "[Error] No queue family with Graphics support found.\n");
            return false;
        }
    } 
    ASSERT_(!settings.requireGrapicsQueues || qfIndex[queue_graphics] != uint32_t(-1));

    if (settings.requireComputeQueues) 
    {
        if (!qfIndices[queue_compute].empty()) 
        {
            qfIndex[queue_compute] = qfIndices[queue_compute][0];
        } 
        else
        {
            if (qfIndex[queue_graphics] != uint32_t(-1)) 
            {
                qfIndex[queue_compute] = qfIndex[queue_graphics];
            } 
            else // settings.requireGrapicsQueues == false
            {
                if (!qfIndices[queue_graphics].empty())
                {
                    qfIndex[queue_compute] = qfIndices[queue_graphics][0];
                } 
                else // qfIndices[queue_compute].empty() && qfIndices[queue_graphics].empty()
                {
                    fprintf(stderr, "[Error] No queue family with Compute support found.\n");
                    return false;
                }
            }
        }
    } 
    ASSERT_(!settings.requireComputeQueues || qfIndex[queue_compute] != uint32_t(-1));

    if (settings.requireTransferQueues) 
    {
        if (!qfIndices[queue_transfer].empty()) 
        {
            qfIndex[queue_transfer] = qfIndices[queue_transfer][0];
        } 
        /*
        The rule of fallback selection:
        - Not requested queue type is preferred to the requested queue type.
        - Graphics queue type is preferred to Compute queue type.
        */
        else
        {
            if (!settings.requireGrapicsQueues && !qfIndices[queue_graphics].empty())
            {
                qfIndex[queue_transfer] = qfIndices[queue_graphics][0];
            }
            else if (!settings.requireComputeQueues && !qfIndices[queue_compute].empty()) 
            {
                qfIndex[queue_transfer] = qfIndices[queue_compute][0];
            } 

            else if (qfIndex[queue_graphics] != uint32_t(-1)) 
            {
                qfIndex[queue_transfer] = qfIndex[queue_graphics];
            } 
            else if (qfIndex[queue_compute] != uint32_t(-1)) 
            {
                qfIndex[queue_transfer] = qfIndex[queue_compute];
            } 
            else // qfIndices[queue_transfer].empty() && qfIndices[queue_graphics].empty() && qfIndices[queue_compute].empty()
            {
                fprintf(stderr, "[Error] No queue family with Transfer support found.\n");
                return false;
            }
        }
    }
    ASSERT_(!settings.requireTransferQueues || qfIndex[queue_transfer] != uint32_t(-1));
    
    std::set<uint32_t> uniqueQfIndices = {
        qfIndex[queue_graphics],
        qfIndex[queue_compute],
        qfIndex[queue_transfer]
    };
    
    std::vector<VkDeviceQueueCreateInfo> queueFamilyInfos;
    std::vector<std::vector<float>> priorities(qfProps.size());
    for (auto qfIndex : uniqueQfIndices) 
    {
        if (qfIndex == uint32_t(-1)) 
            continue;

        priorities[qfIndex].resize(qfProps[qfIndex].queueCount, 0.5f);

        queueFamilyInfos.emplace_back(
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            VkDeviceQueueCreateFlags(0),
            qfIndex,
            qfProps[qfIndex].queueCount,
            priorities[qfIndex].data()     // TODO: Set queue priorities (How to set accross different types but same family?)
        );
    }

    VkPhysicalDeviceSynchronization2Features sync2Features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .synchronization2 = VK_TRUE,
    };
    
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
		.pNext = &sync2Features, 
        .shaderBufferFloat32AtomicAdd = VK_TRUE,// for storageBuffer
        //.shaderSharedFloat32AtomicAdd = VK_TRUE // for shared memory
    };

    VkDeviceCreateInfo deviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &atomicFloatFeatures,
        .queueCreateInfoCount = (uint32_t)queueFamilyInfos.size(),
        .pQueueCreateInfos = queueFamilyInfos.data(),
        .enabledExtensionCount = (uint32_t)reqExtentions.size(),
        .ppEnabledExtensionNames = reqExtentions.data(),
    };
    
    VkDevice vkDevice = create<VkDevice>(physicalDevice, deviceCreateInfo);
    
    std::vector<std::vector<Queue>> queues(qfProps.size());
    for (auto qfIndex : uniqueQfIndices) 
    {
        if (qfIndex == uint32_t(-1)) 
            continue;

        queues[qfIndex].resize(qfProps[qfIndex].queueCount);

        for (uint32_t j = 0; j < qfProps[qfIndex].queueCount; ++j) 
        {
            VkQueue vkQueue;
            vkGetDeviceQueue(vkDevice, qfIndex, j, &vkQueue);
            queues[qfIndex][j].impl = new Queue::Impl(
                vkQueue,
                qfIndex,
                j,
                0.5f);
        }
    }

    auto& device = impl->devices[physicalDevice] = Device();
    device.impl = new Device::Impl(
        physicalDevice,
        vkDevice,
        *this,
        qfIndex[queue_graphics],
        qfIndex[queue_compute],
        qfIndex[queue_transfer],
        std::move(queues)
    );

    return true;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Device
/////////////////////////////////////////////////////////////////////////////////////////
Device::Impl::~Impl() 
{
    vkDeviceWaitIdle(vkDevice);
    
    for (auto qs : queues) for (auto q : qs) delete q.impl;
    for (auto& group : cmdPools) for (auto ele : group.second) delete ele.impl;
    
    for (auto& group : semPools) for (auto ele : group.second) delete ele.impl;
    // for (auto& group : shaderModules) for (auto ele : group.second) delete ele.impl;
    for (auto& group : computePipelines) for (auto ele : group.second) delete ele.impl;
    for (auto& group : buffers) for (auto ele : group.second) delete ele.impl;
    for (auto& group : descSetLayouts) for (auto ele : group.second) delete ele.impl;
    for (auto& group : pipelineLayouts) for (auto ele : group.second) delete ele.impl;
    for (auto& group : descPools) for (auto ele : group.second) delete ele.impl;
    
    vkDestroyDevice(vkDevice, nullptr);
}

void Device::reportGPUQueueFamilies() const
{
    auto qfProps = arrayFrom(vkGetPhysicalDeviceQueueFamilyProperties, impl->vkPhysicalDevice);  
    printf("The device's total queue families:\n");
    for (uint32_t i = 0; i < qfProps.size(); ++i) 
        printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
    fflush(stdout);
}

void Device::reportAssignedQueues() const
{
    printf("Every assigned queues for the logical device:\n");
    auto reportQfs = [&](const std::vector<Queue>& qs) {
        uint32_t i = 0;
        for (auto q : qs) 
        printf("  %u => Family Index: %u, Priority: %.2f\n", 
            i++, 
            q.queueFamilyIndex(),
            q.priority());
    };
    printf("***Graphics Queues***\n");
    reportQfs(impl->queues[impl->qfIndex[queue_graphics]]);
    printf("***Compute Queues***\n");
    reportQfs(impl->queues[impl->qfIndex[queue_compute]]);
    printf("***Transfer Queues***\n");
    reportQfs(impl->queues[impl->qfIndex[queue_transfer]]);
    fflush(stdout);
}

uint32_t Device::queueCount(QueueType type) const 
{ 
    return impl->qCount[type];  // 0 if and only if impl->qfIndex[type] == uint32_t(-1)
}

bool Device::supportPresent(QueueType type) const 
{ 
    return impl->qfIndex[type] != uint32_t(-1) ? false : 
        impl->parent.impl->qfSupportPresent.at(impl->vkPhysicalDevice)[impl->qfIndex[type]];
}

Queue Device::queue(QueueType type, uint32_t index) const 
{ 
    ASSERT_(impl->qfIndex[type] != uint32_t(-1));
    Queue q = impl->queues[impl->qfIndex[type]] [index % impl->qCount[type]];
    q._type = type;
    return q;
}

QueueSelector Device::queue(uint32_t index) const 
{ 
    return QueueSelector(*this, index);
}

CommandPool Device::setDefalutCommandPool(QueueType type, CommandPool cmdPool)
{
    ASSERT_(impl->qfIndex[type] != uint32_t(-1));
    impl->defaultCmdPool[type][cmdPool.impl->flags] = cmdPool;
    return cmdPool;
}

CommandBuffer Device::newCommandBuffer(QueueType type, VkCommandPoolCreateFlags poolFlags)
{
    ASSERT_(impl->qfIndex[type] != uint32_t(-1));
    if (impl->defaultCmdPool[type][poolFlags].impl == nullptr) {
        impl->defaultCmdPool[type][poolFlags] = createCommandPool(type, poolFlags);
    }
    return impl->defaultCmdPool[type][poolFlags].newCommandBuffer();
}


/////////////////////////////////////////////////////////////////////////////////////////
// Queue
/////////////////////////////////////////////////////////////////////////////////////////
uint32_t Queue::queueFamilyIndex() const 
{ 
    return impl->qfIndex; 
}

uint32_t Queue::index() const 
{ 
    return impl->index; 
}

float Queue::priority() const 
{ 
    return impl->priority; 
}

Queue Queue::submit(CommandBuffer cmdBuffer)
{
    // ASSERT_(cmdBuffer.queueFamilyIndex() == impl->qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
    ASSERT_(_type == cmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
    cmdBuffer.impl->lastSubmittedQueue = *this;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer.impl->vkCmdBuffer,
    };
    !vkQueueSubmit(impl->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
    return *this;
}

Queue Queue::submit(std::vector<CommandBuffer> cmdBuffers)
{
    for (auto& cmdBuffer : cmdBuffers) {
        // ASSERT_(cmdBuffer.queueFamilyIndex() == impl->qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
        ASSERT_(_type == cmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
        cmdBuffer.impl->lastSubmittedQueue = *this;
    }

    std::vector<VkCommandBuffer> vkCmdBuffers(cmdBuffers.size());
    for (uint32_t i = 0; i < cmdBuffers.size(); ++i) 
        vkCmdBuffers[i] = cmdBuffers[i].impl->vkCmdBuffer;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = (uint32_t)vkCmdBuffers.size(),
        .pCommandBuffers = vkCmdBuffers.data(),
    };
    !vkQueueSubmit(impl->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
    return *this;
}

Queue Queue::submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence)
{
    size_t batchCount = batches.size();
    size_t waitSemCount = 0;
    size_t cmdBuffCount = 0;
    size_t signalSemCount = 0;
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        waitSemCount += std::get<0>(batches[i]).size();
        cmdBuffCount += std::get<1>(batches[i]).size();
        signalSemCount += std::get<2>(batches[i]).size();
    }

    uint32_t waitSemOffset = 0;
    uint32_t cmdBufferOffset = 0;
    uint32_t signalSemOffset = 0;
    
#ifdef VULKAN_VERSION_1_3
    std::vector<VkSubmitInfo2> submitInfos(batchCount, {VK_STRUCTURE_TYPE_SUBMIT_INFO_2});

    std::vector<VkSemaphoreSubmitInfo> waitSems; waitSems.reserve(waitSemCount);
    std::vector<VkCommandBufferSubmitInfo> cmdBuffers; cmdBuffers.reserve(cmdBuffCount);
    std::vector<VkSemaphoreSubmitInfo> signalSems; signalSems.reserve(signalSemCount);  
    
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        const auto& [inWaitSems, inCmdBuffers, inSignalSems] = batches[i];
        VkSubmitInfo2& info = submitInfos[i];

        info.waitSemaphoreInfoCount = (uint32_t) inWaitSems.size();
        info.pWaitSemaphoreInfos = waitSems.data() + waitSemOffset;

        for (auto& inWaitSem : inWaitSems) 
        {
            waitSems.emplace_back(
                VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                nullptr,
                inWaitSem.sem.impl->vkSemaphore,
                0,
                inWaitSem.stage,
                0
            );
        }
        waitSemOffset += info.waitSemaphoreInfoCount;

        info.commandBufferInfoCount = (uint32_t) inCmdBuffers.size();
        info.pCommandBufferInfos = cmdBuffers.data() + cmdBufferOffset;
        for (auto& inCmdBuffer : inCmdBuffers) 
        {
            // ASSERT_(inCmdBuffer.queueFamilyIndex() == impl->qfIndex); // VUID-vkQueueSubmit2-pCommandBuffers-00074
            ASSERT_(_type == inCmdBuffer.type()); // VUID-vkQueueSubmit2-pCommandBuffers-00074
            inCmdBuffer.impl->lastSubmittedQueue = *this;
            cmdBuffers.emplace_back(
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                nullptr,
                inCmdBuffer.impl->vkCmdBuffer,
                0
            );
        }
        cmdBufferOffset += info.commandBufferInfoCount;

        info.signalSemaphoreInfoCount = (uint32_t) inSignalSems.size();
        info.pSignalSemaphoreInfos = signalSems.data() + signalSemOffset;
        for (auto& inSignalSem : inSignalSems) 
        {
            signalSems.emplace_back(
                VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                nullptr,
                inSignalSem.sem.impl->vkSemaphore,
                0,
                inSignalSem.stage,
                0
            );
        }
        signalSemOffset += info.signalSemaphoreInfoCount;
    }

    !vkQueueSubmit2(impl->vkQueue, batchCount, submitInfos.data(), 
        fence ? fence->impl->vkFence : VK_NULL_HANDLE);

#else
    std::vector<VkSubmitInfo> submitInfos(batchCount, {VK_STRUCTURE_TYPE_SUBMIT_INFO});

    std::vector<VkSemaphore> waitSems; waitSems.reserve(waitSemCount);
    std::vector<VkPipelineStageFlags> waitStages; waitStages.reserve(waitSemCount);
    std::vector<VkCommandBuffer> cmdBuffers; cmdBuffers.reserve(cmdBuffCount);
    std::vector<VkSemaphore> signalSems; signalSems.reserve(signalSemCount);
    
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        const auto& [inWaitSems, inCmdBuffers, inSignalSems] = batches[i];
        VkSubmitInfo& info = submitInfos[i];

        info.waitSemaphoreCount = (uint32_t) inWaitSems.size();
        info.pWaitSemaphores = waitSems.data() + waitSemOffset;
        info.pWaitDstStageMask = waitStages.data() + waitSemOffset;

        for (auto& inWaitSem : inWaitSems) 
        {
            waitSems.push_back(inWaitSem.sem.impl->vkSemaphore);
            waitStages.push_back((VkPipelineStageFlags) inWaitSem.stage);
        }
        waitSemOffset += info.waitSemaphoreCount;

        info.commandBufferCount = (uint32_t) inCmdBuffers.size();
        info.pCommandBuffers = cmdBuffers.data() + cmdBufferOffset;
        for (auto& inCmdBuffer : inCmdBuffers) 
        {
            // ASSERT_(inCmdBuffer.queueFamilyIndex() == impl->qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
            ASSERT_(_type == inCmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
            inCmdBuffer.impl->lastSubmittedQueue = *this;
            cmdBuffers.push_back(inCmdBuffer.impl->vkCmdBuffer);
        }
        cmdBufferOffset += info.commandBufferCount;

        info.signalSemaphoreCount = (uint32_t) inSignalSems.size();
        info.pSignalSemaphores = signalSems.data() + signalSemOffset;
        for (auto& inSignalSem : inSignalSems) 
        {
            signalSems.push_back(inSignalSem.sem.impl->vkSemaphore);
        }
        signalSemOffset += info.signalSemaphoreCount;
    }
    
    !vkQueueSubmit(impl->vkQueue, batchCount, submitInfos.data(), 
        fence ? fence->impl->vkFence : VK_NULL_HANDLE);
#endif

    return *this;
}

Queue Queue::submit(std::vector<SubmissionBatchInfo>&& batches)
{
    return submit(std::move(batches), std::nullopt);
}

Queue Queue::waitIdle()
{
    !vkQueueWaitIdle(impl->vkQueue);
    return *this;
}


/////////////////////////////////////////////////////////////////////////////////////////
// CommandPool
/////////////////////////////////////////////////////////////////////////////////////////
CommandPool Device::createCommandPool(QueueType type, VkCommandPoolCreateFlags flags)
{
    uint32_t qfIndex = impl->qfIndex[type];
    ASSERT_(qfIndex != uint32_t(-1));

    auto vkHandle = create<VkCommandPool>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = flags,
        .queueFamilyIndex = qfIndex,
    });

    CommandPool result;
    result.impl = new CommandPool::Impl(
        impl->vkDevice, 
        *this,
        vkHandle, 
        flags,
        qfIndex,
        type);
    return impl->cmdPools.at(GROUP_PERMANENT).emplace_back(result);
}

QueueType CommandPool::type() const
{
    return impl->type;
}

std::vector<CommandBuffer> CommandPool::newCommandBuffers(uint32_t count)
{
    std::vector<VkCommandBuffer> vkCmdBuffers = allocate<VkCommandBuffer>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = impl->vkCmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,           // TODO: support VK_COMMAND_BUFFER_LEVEL_SECONDARY
        .commandBufferCount = count,
    });

    std::vector<CommandBuffer> cmdBuffers(count);
    for (uint32_t i = 0; i < count; ++i) 
    {
        impl->cmdBuffers.emplace_back(CommandBuffer()).impl = new CommandBuffer::Impl(
            vkCmdBuffers[i], 
            impl->device,
            impl->qfIndex,
            impl->type);
        cmdBuffers[i] = impl->cmdBuffers.back();
    }
    return cmdBuffers;
}

CommandBuffer CommandPool::newCommandBuffer()
{
    return newCommandBuffers(1)[0];
}


/////////////////////////////////////////////////////////////////////////////////////////
// CommandBuffer
/////////////////////////////////////////////////////////////////////////////////////////
QueueType CommandBuffer::type() const
{
    return impl->type;
}

uint32_t CommandBuffer::queueFamilyIndex() const
{
    return impl->qfIndex;
}

CommandBuffer CommandBuffer::submit(uint32_t index) const
{
    impl->device.queue(impl->type, index).submit(*this);
    return *this;
}

Queue CommandBuffer::lastSubmittedQueue() const
{
    return impl->lastSubmittedQueue;
}

CommandBuffer CommandBuffer::begin(VkCommandBufferUsageFlags flags)
{
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = flags,
    };
    !vkBeginCommandBuffer(impl->vkCmdBuffer, &beginInfo);
    return *this;
}

CommandBuffer CommandBuffer::end()
{
    !vkEndCommandBuffer(impl->vkCmdBuffer);
    return *this;
}

CommandBuffer CommandBuffer::bindPipeline(Pipeline pipeline)
{
    impl->boundPipeline = pipeline;

    VkPipelineBindPoint bindPoint;
    VkPipeline vkPipeline;

    std::visit([&](auto&& pipeline) 
    {
        using T = std::decay_t<decltype(pipeline)>;
        if constexpr (std::is_same_v<T, ComputePipeline>) 
        {
            ASSERT_(type() <= queue_compute);  // VUID-vkCmdBindPipeline-pipelineBindPoint-00777
            bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
            vkPipeline = pipeline.impl->vkPipeline;
        }
        else 
            ASSERT_(false); 
    }, pipeline);
    
    vkCmdBindPipeline(impl->vkCmdBuffer, bindPoint, vkPipeline);
    return *this;
}

CommandBuffer CommandBuffer::bindDescSets(
    PipelineLayout layout, 
    VkPipelineBindPoint bindPoint,
    std::vector<DescriptorSet> descSets,
    uint32_t firstSet)
{
    std::vector<VkDescriptorSet> vkDescSets(descSets.size());
    for (uint32_t i = 0; i < descSets.size(); ++i) 
        vkDescSets[i] = descSets[i].impl->vkDescSet;

    vkCmdBindDescriptorSets(
        impl->vkCmdBuffer, bindPoint, 
        layout.impl->vkPipeLayout, firstSet, 
        (uint32_t)vkDescSets.size(), vkDescSets.data(), 
        0, nullptr);
    return *this;
}

CommandBuffer CommandBuffer::bindDescSets(
    std::vector<DescriptorSet> descSets,
    uint32_t firstSet)
{
    std::vector<VkDescriptorSet> vkDescSets(descSets.size());
    for (uint32_t i = 0; i < descSets.size(); ++i) 
        vkDescSets[i] = descSets[i].impl->vkDescSet;

    auto index = impl->boundPipeline.index();
    VkPipelineBindPoint bindPoint;
    VkPipelineLayout layout;
    if (index == 0) {
        bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        layout = std::get<ComputePipeline>(impl->boundPipeline).impl->layout.impl->vkPipeLayout;
    }
    else 
        ASSERT_(false);

    vkCmdBindDescriptorSets(
        impl->vkCmdBuffer, bindPoint, 
        layout, firstSet, 
        (uint32_t)vkDescSets.size(), vkDescSets.data(), 
        0, nullptr);
    return *this;
}

CommandBuffer CommandBuffer::setPushConstants(
    PipelineLayout layout, 
    VkShaderStageFlags stageFlags, 
    uint32_t offset, 
    uint32_t size,
    const void* values)
{
    vkCmdPushConstants(
        impl->vkCmdBuffer, 
        layout.impl->vkPipeLayout, 
        stageFlags, 
        offset, 
        size, 
        values);
    return *this;
}

CommandBuffer CommandBuffer::setPushConstants(
    uint32_t offset, 
    uint32_t size, 
    const void* data)
{
    auto index = impl->boundPipeline.index();
    PipelineLayout layout;
    if (index == 0) {
        layout = std::get<ComputePipeline>(impl->boundPipeline).impl->layout;
    }
    else 
        ASSERT_(false);

    // safe from VUID-vkCmdPushConstants-offset-01796
    VkShaderStageFlags stageFlags = 0;
    for (auto& range: layout.impl->pushConstants)
    {
        if (offset < range.offset + range.size && range.offset < offset + size) 
            stageFlags |= (uint32_t) range.stages;
    }

    vkCmdPushConstants(
        impl->vkCmdBuffer, 
        layout.impl->vkPipeLayout, 
        stageFlags, 
        offset, 
        size, 
        data);
    return *this;
}

CommandBuffer CommandBuffer::barriers(
    std::vector<BarrierInfo> barrierInfos)
{
#ifdef VULKAN_VERSION_1_3
    std::vector<VkMemoryBarrier2> memoryBarriers;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    std::vector<VkImageMemoryBarrier2> imageBarriers;
    bufferBarriers.reserve(barrierInfos.size());
    imageBarriers.reserve(barrierInfos.size());

    for (auto& barrierInfo : barrierInfos) 
    {
        std::visit([&](auto&& barrier) {
            using T = std::decay_t<decltype(barrier)>;

            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            if constexpr (std::is_same_v<T, BufferMemoryBarrier>) 
            {
                // Safe from VUID-vkCmdPipelineBarrier2-srcQueueFamilyIndex-10387 
                if (barrier.opType == OwnershipTransferOpType::release)
                {
                    srcQueueFamilyIndex = queueFamilyIndex();
                    dstQueueFamilyIndex = impl->device.impl->qfIndex[barrier.pairedQueue];
                }
                else if (barrier.opType == OwnershipTransferOpType::acquire)
                {
                    srcQueueFamilyIndex = impl->device.impl->qfIndex[barrier.pairedQueue];
                    dstQueueFamilyIndex = queueFamilyIndex();
                }
                else ASSERT_(barrier.opType == OwnershipTransferOpType::none);
            }
            
            if constexpr (std::is_same_v<T, MemoryBarrier>) 
            {
                memoryBarriers.emplace_back(
                    VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    nullptr,
                    (VkPipelineStageFlags2) barrier.srcMask.stage,
                    (VkAccessFlags2) barrier.srcMask.access,
                    (VkPipelineStageFlags2) barrier.dstMask.stage,
                    (VkAccessFlags2) barrier.dstMask.access
                );
            }
            else if constexpr (std::is_same_v<T, BufferMemoryBarrier>) {
                bufferBarriers.emplace_back(
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    nullptr,
                    (VkPipelineStageFlags2) barrier.srcMask.stage,
                    (VkAccessFlags2) barrier.srcMask.access,
                    (VkPipelineStageFlags2) barrier.dstMask.stage,
                    (VkAccessFlags2) barrier.dstMask.access,
                    srcQueueFamilyIndex,
                    dstQueueFamilyIndex,
                    barrier.buffer.impl->vkBuffer,
                    barrier.offset, 
                    barrier.size
                );
            }
            else if constexpr (std::is_same_v<T, ImageMemoryBarrier>) {
                // TODO: Support image & image barrier
            }
        }, barrierInfo);
    }
    
    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = (uint32_t)memoryBarriers.size(),
        .pMemoryBarriers = memoryBarriers.data(),
        .bufferMemoryBarrierCount = (uint32_t)bufferBarriers.size(),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = (uint32_t)imageBarriers.size(),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(impl->vkCmdBuffer, &depInfo);

#else
    std::vector<VkMemoryBarrier> memoryBarriers;
    std::vector<VkBufferMemoryBarrier> bufferBarriers;
    std::vector<VkImageMemoryBarrier> imageBarriers;
    bufferBarriers.reserve(barrierInfos.size());
    imageBarriers.reserve(barrierInfos.size());

    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;

    for (auto& barrierInfo : barrierInfos) 
    {
        std::visit([&](auto&& barrier) {
            using T = std::decay_t<decltype(barrier)>;

            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            if constexpr (std::is_same_v<T, BufferMemoryBarrier>) 
            {
                // Safe from VUID-vkCmdPipelineBarrier-srcQueueFamilyIndex-00187
                if (barrier.opType == OwnershipTransferOpType::release)
                {
                    srcQueueFamilyIndex = queueFamilyIndex();
                    dstQueueFamilyIndex = impl->device.impl->qfIndex[barrier.pairedQueue];
                }
                else if (barrier.opType == OwnershipTransferOpType::acquire)
                {
                    srcQueueFamilyIndex = impl->device.impl->qfIndex[barrier.pairedQueue];
                    dstQueueFamilyIndex = queueFamilyIndex();
                }
                else ASSERT_(barrier.opType == OwnershipTransferOpType::none);
            }
            
            if constexpr (std::is_same_v<T, MemoryBarrier>) 
            {
                srcStageMask |= (VkPipelineStageFlags) barrier.srcMask.stage;
                dstStageMask |= (VkPipelineStageFlags) barrier.dstMask.stage;
                memoryBarriers.emplace_back(
                    VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    nullptr,
                    (VkAccessFlags) barrier.srcMask.access,
                    (VkAccessFlags) barrier.dstMask.access
                );
            }
            else if constexpr (std::is_same_v<T, BufferMemoryBarrier>) {
                srcStageMask |= (VkPipelineStageFlags) barrier.srcMask.stage;
                dstStageMask |= (VkPipelineStageFlags) barrier.dstMask.stage;
                bufferBarriers.emplace_back(
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    (VkAccessFlags) barrier.srcMask.access,
                    (VkAccessFlags) barrier.dstMask.access,
                    srcQueueFamilyIndex,
                    dstQueueFamilyIndex,
                    barrier.buffer.impl->vkBuffer,
                    barrier.offset, 
                    barrier.size
                );
            }
            else if constexpr (std::is_same_v<T, ImageMemoryBarrier>) {
                // TODO: Support image & image barrier
            }
        }, barrierInfo);
    }
    
    vkCmdPipelineBarrier(
        impl->vkCmdBuffer,
        srcStageMask, dstStageMask,
        0,
        (uint32_t)memoryBarriers.size(), memoryBarriers.data(),
        (uint32_t)bufferBarriers.size(), bufferBarriers.data(),
        (uint32_t)imageBarriers.size(), imageBarriers.data());
#endif

    return *this;
}

CommandBuffer CommandBuffer::barrier(BarrierInfo barrierInfos)
{
    return barriers({barrierInfos});
}

CommandBuffer CommandBuffer::copyBuffer(Buffer dst, Buffer src, uint64_t dstOffset,  uint64_t srcOffset,  uint64_t size)
{
    ASSERT_(srcOffset < src.size()); // VUID-vkCmdCopyBuffer-srcOffset-00113
    ASSERT_(dstOffset < dst.size()); // VUID-vkCmdCopyBuffer-dstOffset-00114

    if (size == VK_WHOLE_SIZE)
        size = std::min(src.size() - srcOffset, dst.size() - dstOffset);  
    else 
    {
        ASSERT_(srcOffset + size <= src.size()); // VUID-vkCmdCopyBuffer-size-00115
        ASSERT_(dstOffset + size <= dst.size()); // VUID-vkCmdCopyBuffer-size-00116
    }

	VkBufferCopy copyRegion{  
		.srcOffset = srcOffset,
		.dstOffset = dstOffset,
		.size = size, 
	};

	vkCmdCopyBuffer(
		impl->vkCmdBuffer,
		src.impl->vkBuffer,
		dst.impl->vkBuffer,
		1, &copyRegion);
	return *this;
}

CommandBuffer CommandBuffer::copyBuffer(BufferRange dst, BufferRange src)
{
    return copyBuffer(
        dst.buffer, src.buffer, 
        dst.offset, src.offset, 
        std::min(src.size, dst.size));
}

CommandBuffer CommandBuffer::dispatch(uint32_t numThreadsInX, uint32_t numThreadsInY, uint32_t numThreadsInZ)
{
    ASSERT_(type() <= queue_compute);  // VUID-vkCmdDispatch-commandBuffer-cmdpool (Implicit)  
    auto pipeline = std::get_if<ComputePipeline>(&impl->boundPipeline);
    ASSERT_(pipeline != nullptr);

    auto [groupSizeInX, groupSizeInY, groupSizeInZ] = pipeline->impl->workGroupSize;

    vkCmdDispatch(
        impl->vkCmdBuffer, 
        (numThreadsInX + groupSizeInX - 1) / groupSizeInX,
        (numThreadsInY + groupSizeInY - 1) / groupSizeInY,
        (numThreadsInZ + groupSizeInZ - 1) / groupSizeInZ);
    return *this;
}

CommandBuffer CommandBuffer::dispatch0(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    ASSERT_(type() <= queue_compute);  // VUID-vkCmdDispatch-commandBuffer-cmdpool (Implicit)  
    vkCmdDispatch(impl->vkCmdBuffer, groupCountX, groupCountY, groupCountZ);
    return *this;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Fence
/////////////////////////////////////////////////////////////////////////////////////////
Fence Device::createFence(VkFenceCreateFlags flags)
{
    auto vkHandle = create<VkFence>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = flags,
    });

    Fence result;
    result.impl = new Fence::Impl(
        impl->vkDevice, 
        vkHandle);
    return impl->fences.at(GROUP_PERMANENT).emplace_back(result);
}

VkResult Device::waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout)
{
    std::vector<VkFence> vkFences(fences.size());
    for (uint32_t i = 0; i < fences.size(); ++i) 
        vkFences[i] = fences[i].impl->vkFence;

    return vkWaitForFences(impl->vkDevice, (uint32_t)fences.size(), vkFences.data(), waitAll, timeout);
}

void Device::resetFences(std::vector<Fence> fences)
{
    std::vector<VkFence> vkFences(fences.size());
    for (uint32_t i = 0; i < fences.size(); ++i) 
        vkFences[i] = fences[i].impl->vkFence;

    !vkResetFences(impl->vkDevice, (uint32_t)fences.size(), vkFences.data());
}

VkResult Fence::wait(uint64_t timeout) const
{
    return vkWaitForFences(impl->vkDevice, 1, &impl->vkFence, VK_TRUE, timeout);
}

void Fence::reset() const
{
    !vkResetFences(impl->vkDevice, 1, &impl->vkFence);
}

bool Fence::isSignaled() const
{
    return vkGetFenceStatus(impl->vkDevice, impl->vkFence) == VK_SUCCESS;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Semaphore
/////////////////////////////////////////////////////////////////////////////////////////
Semaphore Device::createSemaphore()
{
    auto vkHandle = create<VkSemaphore>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    });

    Semaphore result;
    result.impl = new Semaphore::Impl(
        impl->vkDevice, 
        vkHandle);
    return impl->semPools.at(GROUP_PERMANENT).emplace_back(result);
}


/////////////////////////////////////////////////////////////////////////////////////////
// ShaderModule
/////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////
// ComputePipeline
/////////////////////////////////////////////////////////////////////////////////////////
ComputePipeline Device::createComputePipeline(const ComputePipelineCreateInfo& info)
{
    std::vector<uint32_t> spvBinary = glsl2spv(VK_SHADER_STAGE_COMPUTE_BIT, info.csSrc);

    VkShaderModule csModule = create<VkShaderModule>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spvBinary.size() * 4,
        .pCode = spvBinary.data(),
    });

    void* pModule = createReflectShaderModule(spvBinary);
    auto [sizeX, sizeY, sizeZ] = extractWorkGroupSize(pModule);

    PipelineLayout layout = info.layout.has_value() 
        ? info.layout.value() 
        : createPipelineLayout(extractPipelineLayoutDesc(pModule));


    destroyReflectShaderModule(pModule);
    
    VkPipelineLayout vkLayout = layout.impl->vkPipeLayout;

    VkComputePipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = csModule,
            .pName = "main",
        },
        .layout = vkLayout,
    };

    VkPipeline vkHandle;
    !vkCreateComputePipelines(
        impl->vkDevice, VK_NULL_HANDLE, 
        1, &createInfo, 
        nullptr, &vkHandle);
    vkDestroyShaderModule(impl->vkDevice, csModule, nullptr);

    ComputePipeline result;
    result.impl = new ComputePipeline::Impl(
        impl->vkDevice, 
        vkHandle, 
        layout,
        sizeX, sizeY, sizeZ);
    return impl->computePipelines.at(GROUP_PERMANENT).emplace_back(result);
}

PipelineLayout ComputePipeline::layout() const
{
    return impl->layout;
}

DescriptorSetLayout ComputePipeline::descSetLayout(uint32_t setId) const
{
    return impl->layout.descSetLayout(setId);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Buffer
/////////////////////////////////////////////////////////////////////////////////////////
Buffer Device::createBuffer(const BufferCreateInfo& info)
{
    auto vkHandle = create<VkBuffer>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = info.size,
        .usage = info.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,  // TODO: support VK_SHARING_MODE_CONCURRENT
    });

    auto memInfo = getMemoryAllocInfo(
        impl->vkPhysicalDevice, impl->vkDevice, vkHandle, info.reqMemProps);
    
    VkDeviceMemory memory = allocate<VkDeviceMemory>(impl->vkDevice, memInfo.first);
    !vkBindBufferMemory(impl->vkDevice, vkHandle, memory, 0);

    Buffer result;
    result.impl = new Buffer::Impl(
        impl->vkDevice, 
        vkHandle, 
        memory, 
        info.size, 
        info.usage,
        info.reqMemProps,
        memInfo.second);
    return impl->buffers.at(GROUP_PERMANENT).emplace_back(result);
}

uint8_t* Buffer::map(uint64_t offset, uint64_t size)
{
    ASSERT_(!impl->mapped);                                         // VUID-vkMapMemory-memory-00678        
    ASSERT_(impl->memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);  // VUID-vkMapMemory-memory-00682
    ASSERT_(offset < impl->size);                                   // VUID-vkMapMemory-offset-00679
    ASSERT_(size == VK_WHOLE_SIZE || offset + size <= impl->size);  // VUID-vkMapMemory-size-00681

    !vkMapMemory(impl->vkDevice, impl->vkMemory, offset, size, 0, (void**)&impl->mapped);
    impl->mappedOffset = offset;
    impl->mappedSize = size == VK_WHOLE_SIZE ? impl->size : size;
    return impl->mapped;
}

VkMappedMemoryRange Buffer::Impl::getRange(uint64_t offset, uint64_t size)
{
    ASSERT_(mapped);                                                                    // VUID-VkMappedMemoryRange-memory-00684
    if (size == VK_WHOLE_SIZE) 
        ASSERT_(mappedOffset <= offset && offset <= mappedOffset + mappedSize);         // VUID-VkMappedMemoryRange-memory-00686
    else 
        ASSERT_(mappedOffset <= offset && offset + size <= mappedOffset + mappedSize);  // VUID-VkMappedMemoryRange-memory-00685

    return {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory,
        .offset = offset,
        .size = size,
    };
}

void Buffer::flush(uint64_t offset, uint64_t size) const
{
    VkMappedMemoryRange range = impl->getRange(offset, size);
    if (impl->memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) 
        return;
    vkFlushMappedMemoryRanges(impl->vkDevice, 1, &range);
}

void Buffer::invalidate(uint64_t offset, uint64_t size) const
{
    VkMappedMemoryRange range = impl->getRange(offset, size);
    if (impl->memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) 
        return;
    vkInvalidateMappedMemoryRanges(impl->vkDevice, 1, &range);
}

void Buffer::unmap()
{
    ASSERT_(impl->mapped); // VUID-vkUnmapMemory-memory-00689
    ASSERT_(impl->memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    vkUnmapMemory(impl->vkDevice, impl->vkMemory);
    impl->mapped = nullptr;
    impl->mappedOffset = 0;
    impl->mappedSize = 0;
}

uint64_t Buffer::size() const
{
    return impl->size;
}

VkBufferUsageFlags Buffer::usage() const
{
    return impl->usage;
}

VkMemoryPropertyFlags Buffer::memoryProperties() const
{
    return impl->memProps;
}

VkDescriptorBufferInfo Buffer::descInfo(uint64_t offset, uint64_t size) const
{
    return {
        .buffer = impl->vkBuffer,
        .offset = offset,
        .range = size,
    };
}

BufferMemoryBarrier Buffer::barrier(
    PIPELINE_STAGE_ACCESS srcMask,
    PIPELINE_STAGE_ACCESS dstMask,
    OwnershipTransferOpType opType,
    QueueType queueType,
    uint64_t offset,
    uint64_t size
) const
{
    return {
        .srcMask = srcMask,
        .dstMask = dstMask,
        .opType = opType,
        .pairedQueue = queueType,
        .buffer = *this,
        .offset = offset,
        .size = size,
    };
} 

BufferRange Buffer::operator()(uint64_t offset, uint64_t size) const
{    
    if (size == VK_WHOLE_SIZE) 
    {
        ASSERT_(offset < impl->size); 
        size = impl->size - offset;
    }
    else 
        ASSERT_(offset + size <= impl->size);
    
    return {*this, offset, size};
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorSetLayout
/////////////////////////////////////////////////////////////////////////////////////////
DescriptorSetLayout Device::createDescriptorSetLayout(const DescriptorSetLayoutDesc& desc)
{
    std::vector<VkDescriptorSetLayoutBinding> vkBindings; vkBindings.reserve(desc.bindings.size());

    for (const auto& bindingInfo : desc.bindings) 
    {
        vkBindings.emplace_back(
            bindingInfo.binding,
            (VkDescriptorType)(uint32_t)bindingInfo.type,
            bindingInfo.count,
            bindingInfo.stages == SHADER_STAGE::NONE ? 
                VK_SHADER_STAGE_ALL : 
                (VkShaderStageFlags)(uint32_t)bindingInfo.stages,
            nullptr // TODO: support immutable samplers
        );
    }
    
    auto vkHandle = create<VkDescriptorSetLayout>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)vkBindings.size(),
        .pBindings = vkBindings.data(),
    });

    DescriptorSetLayout result;
    result.impl = new DescriptorSetLayout::Impl(
        impl->vkDevice,
        vkHandle,
        std::move(vkBindings)
    );
    return impl->descSetLayouts.at(GROUP_PERMANENT).emplace_back(result);
}

DescriptorSet DescriptorSetLayout::newDescSet(DescriptorPool pool)
{
    return pool.newDescSets({*this})[0];
}

const VkDescriptorSetLayoutBinding& DescriptorSetLayout::bindingInfo(uint32_t bindingId, bool exact) const
{
    auto iter = exact ? impl->bindingInfos.find(bindingId)
        : impl->bindingInfos.lower_bound(bindingId);

    ASSERT_(iter != impl->bindingInfos.end());
    return iter->second;
}


/////////////////////////////////////////////////////////////////////////////////////////
// PipelineLayout
/////////////////////////////////////////////////////////////////////////////////////////
PipelineLayout Device::createPipelineLayout(const PipelineLayoutDesc& desc)
{
    std::vector<DescriptorSetLayout> setLayouts; setLayouts.reserve(desc.setLayouts.size());
    std::vector<VkDescriptorSetLayout> vkSetLayouts; vkSetLayouts.reserve(desc.setLayouts.size());
    std::vector<VkPushConstantRange> vkPushConstants; vkPushConstants.reserve(desc.pushConstants.size());
    
    for (const auto& setLayoutInfo : desc.setLayouts) 
    {
        if (auto* setLayout = std::get_if<DescriptorSetLayout>(&setLayoutInfo)) 
        {
            setLayouts.push_back(*setLayout);
        }
        else if (auto* setLayoutDesc = std::get_if<DescriptorSetLayoutDesc>(&setLayoutInfo)) 
        {
            setLayouts.push_back(createDescriptorSetLayout(std::move(*setLayoutDesc)));
        }
        vkSetLayouts.push_back(setLayouts.back().impl->vkSetLayout);
    }

    for (const auto& pushConstant : desc.pushConstants) 
    {
        vkPushConstants.emplace_back(
            pushConstant.stages == SHADER_STAGE::NONE ? 
                VK_SHADER_STAGE_ALL : 
                (VkShaderStageFlags)(uint32_t)pushConstant.stages,
            pushConstant.offset,
            pushConstant.size
        );
    }
    
    auto vkHandle = create<VkPipelineLayout>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = (uint32_t)vkSetLayouts.size(),
        .pSetLayouts = vkSetLayouts.data(),
        .pushConstantRangeCount = (uint32_t)vkPushConstants.size(),
        .pPushConstantRanges = vkPushConstants.data(),
    });

    PipelineLayout result;
    result.impl = new PipelineLayout::Impl(
        impl->vkDevice,
        vkHandle,
        std::move(setLayouts),
        desc.pushConstants
    );
    return impl->pipelineLayouts.at(GROUP_PERMANENT).emplace_back(result);
}

DescriptorSetLayout PipelineLayout::descSetLayout(uint32_t setId) const
{
    ASSERT_(setId < impl->setLayouts.size()); 
    return impl->setLayouts[setId];
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorPool
/////////////////////////////////////////////////////////////////////////////////////////
DescriptorPool Device::createDescriptorPool(const DescriptorPoolCreateInfo& info)
{
    auto vkHandle = create<VkDescriptorPool>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = info.maxSets,
        .poolSizeCount = (uint32_t)info.maxTypes.size(),
        .pPoolSizes = info.maxTypes.data(),
    });

    DescriptorPool result;
    result.impl = new DescriptorPool::Impl(
        impl->vkDevice,
        vkHandle);
    return impl->descPools.at(GROUP_PERMANENT).emplace_back(result);
}

std::vector<DescriptorSet> DescriptorPool::newDescSets(std::vector<DescriptorSetLayout> setLayouts)
{
    std::vector<VkDescriptorSetLayout> vkSetLayouts(setLayouts.size());
    for (uint32_t i = 0; i < setLayouts.size(); ++i) 
        vkSetLayouts[i] = setLayouts[i].impl->vkSetLayout;

    std::vector<VkDescriptorSet> vkDescSets = allocate<VkDescriptorSet>(impl->vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = impl->vkDescPool,
        .descriptorSetCount = (uint32_t)vkSetLayouts.size(),
        .pSetLayouts = vkSetLayouts.data(),
    });

    std::vector<DescriptorSet> descSets(setLayouts.size());
    for (uint32_t i = 0; i < setLayouts.size(); ++i) 
    {
        impl->descSets.emplace_back(DescriptorSet()).impl = new DescriptorSet::Impl(
            impl->vkDevice, 
            vkDescSets[i], 
            setLayouts[i]);
            
        descSets[i] = impl->descSets.back();
    }
    return descSets;
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorSet
/////////////////////////////////////////////////////////////////////////////////////////
DescriptorSet DescriptorSet::write(std::vector<Buffer> data, uint32_t startBindingId, uint32_t startArrayOffset)
{
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(data.size());
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(data.size());

    auto& bindingInfos = impl->layout.impl->bindingInfos;
    uint32_t dataConsumed = 0;
    
    auto iter = bindingInfos.lower_bound(startBindingId);   // It avoids VUID-VkWriteDescriptorSet-dstBinding-00316
    ASSERT_(startArrayOffset == 0 || 
        (iter->first == startBindingId && startArrayOffset < iter->second.descriptorCount)); // Undefined behavior, not even mentioned in the Vulkan spec.

    while(iter != bindingInfos.end()) 
    {
        auto iter0 = iter;
        auto& headInfo = iter0->second;
        uint32_t consecutiveDescCount = headInfo.descriptorCount - startArrayOffset;
        VkDescriptorType descriptorType = headInfo.descriptorType;
        VkShaderStageFlags stageFlags = headInfo.stageFlags;

        while (++iter != bindingInfos.end()        // See "Consecutive Binding Updates" in Vulkan spec.
            && iter->second.stageFlags == stageFlags
            && iter->second.descriptorType == descriptorType) 
        {
            consecutiveDescCount += iter->second.descriptorCount;
        } 
        consecutiveDescCount = std::min(consecutiveDescCount, (uint32_t)data.size() - dataConsumed);

        for (uint32_t i = 0; i < consecutiveDescCount; ++i) 
            bufferInfos.push_back(data[dataConsumed + i].descInfo());

        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = impl->vkDescSet,
            .dstBinding = iter0->first,
            .dstArrayElement = startArrayOffset,
            .descriptorCount = consecutiveDescCount,
            .descriptorType = descriptorType,
            .pBufferInfo = bufferInfos.data() + dataConsumed,
        });

        dataConsumed += consecutiveDescCount;
        startArrayOffset = 0;

        if (dataConsumed == data.size()) // Normal exit condition
            break;
    }
    ASSERT_(dataConsumed == data.size()); // If given input data has not been fully consumed, it is considered as an error.
    
    vkUpdateDescriptorSets(impl->vkDevice, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    return *this;
}
