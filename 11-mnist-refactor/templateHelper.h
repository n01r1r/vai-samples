#ifndef TEMPLATE_HELPER_H
#define TEMPLATE_HELPER_H

#include <vulkan/vulkan_core.h>
#include <vector>
#include "error.h"


template <typename T>
struct FunctionTraits;

template <typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)> {
    using ReturnType = R;
    using ArgTuple = std::tuple<Args...>;
    static constexpr size_t ArgCount = sizeof...(Args);
    
    template <size_t N>
    static auto SFINAE() {
        if constexpr (N <= ArgCount) {
            return std::tuple_element_t<ArgCount - N, ArgTuple>{};
        }
        else {
            return std::nullptr_t{};
        }
    }

    using LastArgType = decltype(SFINAE<1>());
    using LastSecondArgType = decltype(SFINAE<2>());
    using LastThirdArgType = decltype(SFINAE<3>());
};




template <typename F, typename... Args>
inline auto arrayFrom(F pFunc, Args... args)
{
    using T = std::remove_pointer_t<typename FunctionTraits<F>::LastArgType>;
    uint32_t count;
    pFunc(args..., &count, static_cast<T*>(nullptr));
    std::vector<T> result(count);
    pFunc(args..., &count, result.data());
    return result;
}




template <typename HandleType>
struct VKHandleFactory;

#define VK_HANDLE_FACTORY_INCLUDE(HandleType, factoryFunc) \
template <> \
struct VKHandleFactory<HandleType> { \
    static constexpr auto f = factoryFunc; \
    using info_t = std::remove_pointer_t< std::remove_pointer_t< \
        typename FunctionTraits< decltype(&factoryFunc) >::LastThirdArgType \
    > >; \
};
VK_HANDLE_FACTORY_INCLUDE(VkInstance, vkCreateInstance);
VK_HANDLE_FACTORY_INCLUDE(VkDevice, vkCreateDevice);
VK_HANDLE_FACTORY_INCLUDE(VkCommandPool, vkCreateCommandPool);
VK_HANDLE_FACTORY_INCLUDE(VkFence, vkCreateFence);
VK_HANDLE_FACTORY_INCLUDE(VkSemaphore, vkCreateSemaphore);
VK_HANDLE_FACTORY_INCLUDE(VkShaderModule, vkCreateShaderModule);
VK_HANDLE_FACTORY_INCLUDE(VkBuffer, vkCreateBuffer);
VK_HANDLE_FACTORY_INCLUDE(VkImage, vkCreateImage);
VK_HANDLE_FACTORY_INCLUDE(VkDescriptorSetLayout, vkCreateDescriptorSetLayout);
VK_HANDLE_FACTORY_INCLUDE(VkPipelineLayout, vkCreatePipelineLayout);
VK_HANDLE_FACTORY_INCLUDE(VkDescriptorPool, vkCreateDescriptorPool);


template <typename HandleType>
auto create(
    const typename VKHandleFactory<HandleType>::info_t& ci,
    const VkAllocationCallbacks* pAllocator = nullptr)
{
    HandleType handle;
    !VKHandleFactory<HandleType>::f(&ci, pAllocator, &handle);
    return handle;
}

template <typename HandleType, typename Parent>
auto create(
    Parent parent, 
    const typename VKHandleFactory<HandleType>::info_t& ci,
    const VkAllocationCallbacks* pAllocator = nullptr) 
{
    HandleType handle;
    !VKHandleFactory<HandleType>::f(parent, &ci, pAllocator, &handle);
    return handle;
}


// VK_HANDLE_FACTORY_INCLUDE(VkDeviceMemory, vkAllocateMemory);

// template <typename HandleType, typename Parent>
// auto allocate(
//     Parent parent, 
//     const typename VKHandleFactory<HandleType>::info_t& ci,
//     const VkAllocationCallbacks* pAllocator = nullptr) 
// {
//     HandleType handle;
//     !VKHandleFactory<HandleType>::f(parent, &ci, pAllocator, &handle);
//     return handle;
// }


template <typename HandleType>
struct VKHandleFactory2;

#define VK_HANDLE_FACTORY_INCLUDE2(HandleType, factoryFunc, countAtOnce) \
template <> \
struct VKHandleFactory2<HandleType> { \
    static constexpr auto f = factoryFunc; \
    using info_t = std::remove_pointer_t< std::remove_pointer_t< \
        typename FunctionTraits< decltype(&factoryFunc) >::LastSecondArgType \
    > >; \
    static constexpr auto outputSize = &info_t::countAtOnce; \
};
VK_HANDLE_FACTORY_INCLUDE2(VkCommandBuffer, vkAllocateCommandBuffers, commandBufferCount);
VK_HANDLE_FACTORY_INCLUDE2(VkDescriptorSet, vkAllocateDescriptorSets, descriptorSetCount);


template <typename HandleType, typename Parent>
std::vector<HandleType> allocate(
    Parent parent, 
    const typename VKHandleFactory2<HandleType>::info_t& ci) 
{
    uint32_t size = ci.*VKHandleFactory2<HandleType>::outputSize;
    std::vector<HandleType> handles(size);
    !VKHandleFactory2<HandleType>::f(parent, &ci, handles.data());
    return handles;
}

template <typename HandleType>
std::enable_if_t<std::is_same_v<HandleType, VkDeviceMemory>, VkDeviceMemory>
allocate(
    VkDevice parent, 
    const VkMemoryAllocateInfo& ci,
    const VkAllocationCallbacks* pAllocator = nullptr) 
{
    VkDeviceMemory handle;
    !vkAllocateMemory(parent, &ci, pAllocator, &handle);
    return handle;
}
#endif // TEMPLATE_HELPER_H
