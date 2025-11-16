#ifndef VULKAN_APP_H
#define VULKAN_APP_H

#include <vulkan/vulkan_core.h>
#include <vector>
#include <string>
#include <variant>
#include <optional>
// #include <tuple>
#define VULKAN_VERSION_1_3  // TODO: whether to use this or not depends on the system

namespace vk {

class VulkanApp;
class Device;
class Queue;
class CommandPool;
class CommandBuffer;
class Fence;
class Semaphore;

class ShaderModule;
class ComputePipeline;
class GraphicsPipeline;
class RaytracingPipeline;

class Buffer;
//class Image;

class DescriptorSetLayout;
class PipelineLayout;
class DescriptorPool;
class DescriptorSet;


#define VULKAN_CLASS_COMMON \
friend class VulkanApp; \
friend class Device; \
friend class Queue; \
friend class CommandPool; \
friend class CommandBuffer; \
friend class Fence; \
friend class Semaphore; \
friend class ShaderModule; \
friend class ComputePipeline; \
friend class GraphicsPipeline; \
friend class RaytracingPipeline; \
friend class Buffer; \
friend class DescriptorSetLayout; \
friend class PipelineLayout; \
friend class DescriptorPool; \
friend class DescriptorSet; \
friend class Submitting; \
class Impl; Impl* impl; \
public: \
operator bool() const { return impl != nullptr; } \
private: \


#define VULKAN_CLASS_COMMON2 \
friend class VulkanApp; \
friend class Device; \
friend class Queue; \
friend class CommandPool; \
friend class CommandBuffer; \
friend class Fence; \
friend class Semaphore; \
friend class ShaderModule; \
friend class ComputePipeline; \
friend class GraphicsPipeline; \
friend class RaytracingPipeline; \
friend class Buffer; \
friend class DescriptorSetLayout; \
friend class PipelineLayout; \
friend class DescriptorPool; \
friend class DescriptorSet; \
friend class Submitting; \
class Impl; 


struct ComputePipelineCreateInfo;
struct BufferCreateInfo;
struct DescriptorPoolCreateInfo;
struct BufferRange;
struct SemaphoreStage;
struct QueueSelector;
struct BindingInfo;
struct DescriptorSetLayoutDesc;
struct PipelineLayoutDesc;

struct MemoryBarrier;
struct BufferMemoryBarrier;
struct ImageMemoryBarrier;

struct DESCRIPTOR_TYPE;
struct SHADER_STAGE;
struct PIPELINE_STAGE;
struct ACCESS;
struct PIPELINE_STAGE_ACCESS;


using BarrierInfo = std::variant<MemoryBarrier, BufferMemoryBarrier, ImageMemoryBarrier>;
using Pipeline = std::variant<ComputePipeline, GraphicsPipeline, RaytracingPipeline>;
using Integer = std::variant<int, uint32_t>;
using SubmissionBatchInfo = std::tuple<
    std::vector<SemaphoreStage>, 
    std::vector<CommandBuffer>, 
    std::vector<SemaphoreStage>
>; 

struct DeviceSettings {
    bool requireGrapicsQueues = true;
    bool requireComputeQueues = true;
    bool requireTransferQueues = true;
    bool supportPresent = false;
    bool supportRaytracing = false;
};


enum QueueType {
    queue_graphics, 
    queue_compute, 
    queue_transfer, 
    queue_max,
};


enum class OwnershipTransferOpType {
    none,
    release,
    acquire,
};


class VulkanApp {
    VULKAN_CLASS_COMMON
    ~VulkanApp();
    VulkanApp(DeviceSettings defaultSettings = {});
    VulkanApp(const VulkanApp&) = delete;
    VulkanApp& operator=(const VulkanApp&) = delete;

public:
    static VulkanApp& get();    // singleton pattern
    bool initDevice(uint32_t gpuIndex, DeviceSettings settings={});
    Device device(uint32_t gpuIndex=uint32_t(-1));
};


class Device {
    VULKAN_CLASS_COMMON

public:
    void reportGPUQueueFamilies() const;
    void reportAssignedQueues() const;    

    uint32_t queueCount(QueueType type) const;
    bool supportPresent(QueueType type) const;
    Queue queue(QueueType type, uint32_t index=0) const;
    QueueSelector queue(uint32_t index=0) const;

    CommandPool createCommandPool(QueueType type, VkCommandPoolCreateFlags flags=0);
    CommandPool setDefalutCommandPool(QueueType type, CommandPool cmdPool);
    CommandBuffer newCommandBuffer(QueueType type, VkCommandPoolCreateFlags poolFlags=0);

    Fence createFence(VkFenceCreateFlags flags=0);
    VkResult waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout=uint64_t(-1));
    void resetFences(std::vector<Fence> fences);
    Semaphore createSemaphore();

    // ShaderModule createShaderModule(ShaderModuleCreateInfo info);
    ComputePipeline createComputePipeline(const ComputePipelineCreateInfo& info);
    Buffer createBuffer(const BufferCreateInfo& info);
    DescriptorSetLayout createDescriptorSetLayout(const DescriptorSetLayoutDesc& desc);
    PipelineLayout createPipelineLayout(const PipelineLayoutDesc& desc);
    DescriptorPool createDescriptorPool(const DescriptorPoolCreateInfo& info);

    void destroyCommandPools(std::string groupName);
    void destroyFences(std::string groupName);
    void destroySemaphores(std::string groupName);
    void destroyComputePipelines(std::string groupName);
    void destroyBuffers(std::string groupName);
    void destroyDescriptorSetLayouts(std::string groupName);
    void destroyPipelineLayouts(std::string groupName);
    void destroyDescriptorPools(std::string groupName);
};


class Queue {
    VULKAN_CLASS_COMMON
    QueueType _type = queue_max;
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    uint32_t index() const;

    float priority() const;

    Queue submit(
        CommandBuffer cmdBuffer
    );

    Queue submit(
        std::vector<CommandBuffer> cmdBuffers
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches,
        std::optional<Fence> fence
    );

    Queue waitIdle();
};


class CommandPool {
    VULKAN_CLASS_COMMON
public:

    QueueType type() const;

    std::vector<CommandBuffer> newCommandBuffers(
        uint32_t count
    );

    CommandBuffer newCommandBuffer();
};


class CommandBuffer {
    VULKAN_CLASS_COMMON
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    CommandBuffer submit(uint32_t index=0) const;

    Queue lastSubmittedQueue() const;
    
    void wait() const {
        lastSubmittedQueue().waitIdle();
    }

    CommandBuffer begin(
        VkCommandBufferUsageFlags flags=0
    );

    CommandBuffer end();

    CommandBuffer bindPipeline(
        Pipeline pipeline
    );

    CommandBuffer bindDescSets(
        PipelineLayout layout, 
        VkPipelineBindPoint bindPoint,
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer bindDescSets(
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer setPushConstants(
        PipelineLayout layout, 
        VkShaderStageFlags stageFlags, 
        uint32_t offset, 
        uint32_t size,
        const void* values
    );

    CommandBuffer setPushConstants(
        uint32_t offset, 
        uint32_t size, 
        const void* data
    );

    CommandBuffer barriers(
        std::vector<BarrierInfo> barrierInfos
    );

    CommandBuffer barrier(
        BarrierInfo barrierInfos
    );

	CommandBuffer copyBuffer(
        Buffer dst, 
        Buffer src, 
        uint64_t dstOffset = 0, 
        uint64_t srcOffset = 0, 
        uint64_t size = VK_WHOLE_SIZE
    );

    CommandBuffer copyBuffer(
        BufferRange dst, 
        BufferRange src
    );

    CommandBuffer dispatch(
        uint32_t numThreadsInX, 
        uint32_t numThreadsInY=1, 
        uint32_t numThreadsInZ=1
    );

    CommandBuffer dispatch0(
        uint32_t groupCountX, 
        uint32_t groupCountY=1, 
        uint32_t groupCountZ=1);
};


class Fence {
    VULKAN_CLASS_COMMON
public:

    VkResult wait(
        uint64_t timeout=uint64_t(-1)
    ) const;

    void reset() const;

    bool isSignaled() const;
};


class Semaphore {
    VULKAN_CLASS_COMMON
public:

    SemaphoreStage operator[](
        PIPELINE_STAGE stage
    ) const;

};


// class ShaderModule {
//     VULKAN_CLASS_COMMON
// public:
//     VkShaderStageFlags stage() const;
// };


class GraphicsPipeline {};
class RaytracingPipeline {};

class ComputePipeline {
    VULKAN_CLASS_COMMON
public:

    PipelineLayout layout() const;

    DescriptorSetLayout descSetLayout(
        uint32_t setId0=0
    ) const;
};



class Buffer {
    VULKAN_CLASS_COMMON
public:
    
    uint8_t* map(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    );
 
    void flush(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;

    void invalidate(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;
    
    void unmap();
    
    uint64_t size() const;

    VkBufferUsageFlags usage() const;

    VkMemoryPropertyFlags memoryProperties() const;

    VkDescriptorBufferInfo descInfo(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;

    BufferMemoryBarrier barrier(
        PIPELINE_STAGE_ACCESS srcMask,
        PIPELINE_STAGE_ACCESS dstMask,
        OwnershipTransferOpType opType=OwnershipTransferOpType::none,
        QueueType queueType=queue_max,
        uint64_t offset=0,
        uint64_t size=VK_WHOLE_SIZE
    ) const;

    BufferRange operator()(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;
};


class DescriptorSetLayout {
    VULKAN_CLASS_COMMON
public:

    DescriptorSet newDescSet(
        DescriptorPool pool
    );

    const VkDescriptorSetLayoutBinding& bindingInfo(
        uint32_t bindingId, 
        bool exact=true
    ) const;
        
    // const std::map<uint32_t, VkDescriptorSetLayoutBinding>& bindingInfos() const;
};


class PipelineLayout {
    VULKAN_CLASS_COMMON
public:

    DescriptorSetLayout descSetLayout(
        uint32_t setId
    ) const;
};


class DescriptorPool {
    VULKAN_CLASS_COMMON
public:

    std::vector<DescriptorSet> newDescSets(
        std::vector<DescriptorSetLayout> layouts
    );
};


class DescriptorSet {
    VULKAN_CLASS_COMMON
public:

    DescriptorSet write(
        std::vector<Buffer> data, 
        uint32_t startBindingId=0, 
        uint32_t startArrayOffset=0
    );
};




///////////////////////////////////////////////////////////////////////////////

struct SHADER_STAGE {
    struct T {
        uint32_t flag;
        T(uint32_t flag) : flag(flag) {}
        SHADER_STAGE operator|(T other) const { 
            return {flag | other.flag}; 
        }
    } flags;

    SHADER_STAGE(T flag) : flags(flag) {}

    SHADER_STAGE(uint32_t flag) : flags(T(flag)) {}
    
    SHADER_STAGE operator|(SHADER_STAGE other) const { 
        return {flags | other.flags}; 
    }

    bool operator==(SHADER_STAGE other) const { 
        return flags.flag == other.flags.flag; 
    }

    operator uint32_t() const { 
        return flags.flag; 
    }

    inline static const T NONE                      = 0x00000000;
    inline static const T VERTEX                    = 0x00000001;
    inline static const T TESSELLATION_CONTROL      = 0x00000002;
    inline static const T TESSELLATION_EVALUATION   = 0x00000004;
    inline static const T GEOMETRY                  = 0x00000008;
    inline static const T FRAGMENT                  = 0x00000010;
    inline static const T ALL_GRAPHICS              = 0x0000001F;
    inline static const T COMPUTE                   = 0x00000020;
    inline static const T TASK                      = 0x00000040;
    inline static const T MESH                      = 0x00000080;
    inline static const T RAYGEN                    = 0x00000100;
    inline static const T ANY_HIT                   = 0x00000200;
    inline static const T CLOSEST_HIT               = 0x00000400;
    inline static const T MISS                      = 0x00000800;
    inline static const T INTERSECTION              = 0x00001000;
    inline static const T CALLABLE                  = 0x00002000;
    inline static const T ALL                       = 0x7FFFFFFF;
};


struct DESCRIPTOR_TYPE {
    struct T {
        uint32_t id;
        T(uint32_t id) : id(id) {}
        BindingInfo operator[](uint32_t count) const;
    } id;

    DESCRIPTOR_TYPE(T id) : id(id) {}

    DESCRIPTOR_TYPE(uint32_t id) : id(T(id)) {}

    operator uint32_t() const { 
        return id.id; 
    }

    inline static const T SAMPLER = 0;
    inline static const T COMBINED_IMAGE_SAMPLER = 1;
    inline static const T SAMPLED_IMAGE = 2;
    inline static const T STORAGE_IMAGE = 3;
    inline static const T UNIFORM_TEXEL_BUFFER = 4;
    inline static const T STORAGE_TEXEL_BUFFER = 5;
    inline static const T UNIFORM_BUFFER = 6;
    inline static const T STORAGE_BUFFER = 7;
    inline static const T UNIFORM_BUFFER_DYNAMIC = 8;
    inline static const T STORAGE_BUFFER_DYNAMIC = 9;
    inline static const T INPUT_ATTACHMENT = 10;
    inline static const T INLINE_UNIFORM_BLOCK                  = 1000138000;
    inline static const T ACCELERATION_STRUCTURE_KHR            = 1000150000;
    inline static const T MUTABLE_EXT                           = 1000351000;
    inline static const T SAMPLE_WEIGHT_IMAGE_QCOM              = 1000440000;
    inline static const T BLOCK_MATCH_IMAGE_QCOM                = 1000440001;
    inline static const T PARTITIONED_ACCELERATION_STRUCTURE_NV = 1000570000;
    inline static const T MAX_ENUM                              = 0x7FFFFFFF;
};


struct BindingInfo {
    uint32_t binding;
    DESCRIPTOR_TYPE type;
    uint32_t count = 1;
    SHADER_STAGE stages = SHADER_STAGE::NONE;
};


inline BindingInfo DESCRIPTOR_TYPE::T::operator[](uint32_t count) const
{
    return {
        .type = *this,
        .count = count
    };
}

inline BindingInfo operator<=>(uint32_t slot, DESCRIPTOR_TYPE type)
{
    return {
        .binding = slot,
        .type = type
    };
}

inline BindingInfo&& operator<=>(uint32_t slot, BindingInfo&& bindingInfo)
{
    bindingInfo.binding = slot;
    return std::move(bindingInfo);
}

inline BindingInfo&& operator|(BindingInfo&& bindingInfo, SHADER_STAGE stages)
{
    bindingInfo.stages = bindingInfo.stages | stages;
    return std::move(bindingInfo);
}


struct DescriptorSetLayoutDesc {
    std::vector<BindingInfo> bindings;

    DescriptorSetLayoutDesc() = default;

    DescriptorSetLayoutDesc(const DescriptorSetLayoutDesc&) = default;
    
    DescriptorSetLayoutDesc(DescriptorSetLayoutDesc&& other) = default;

    DescriptorSetLayoutDesc(BindingInfo&& info)
    : bindings{std::move(info)} {}
};


using DescriptorSetLayoutInfo = std::variant<
    DescriptorSetLayout,  
    DescriptorSetLayoutDesc>;


inline DescriptorSetLayoutDesc operator,(DescriptorSetLayoutDesc&& desc, BindingInfo&& info)
{
    desc.bindings.push_back(std::move(info));
    return std::move(desc);
}

inline DescriptorSetLayoutDesc operator|(DescriptorSetLayoutDesc&& desc, SHADER_STAGE stages)
{
    for (auto& binding : desc.bindings) {
        binding.stages = binding.stages | stages;
    }
    return std::move(desc);
}

#define EmptySet DescriptorSetLayoutDesc{}


struct PushConstantRange {
    SHADER_STAGE stages = SHADER_STAGE::NONE;
    uint32_t offset = 0;
    uint32_t size;

    PushConstantRange(uint32_t offset, uint32_t size)
    : offset(offset), size(size) {}

    PushConstantRange(uint32_t size)
    : size(size) {}

    PushConstantRange&& operator|(SHADER_STAGE stages) &&
    {
        this->stages = this->stages | stages;
        return std::move(*this);
    }
};


struct PipelineLayoutDesc {
    std::vector<DescriptorSetLayoutInfo> setLayouts;
    std::vector<PushConstantRange> pushConstants;

    PipelineLayoutDesc() = default;

    PipelineLayoutDesc(const PipelineLayoutDesc&) = default;
    
    PipelineLayoutDesc(PipelineLayoutDesc&& other) = default;

    PipelineLayoutDesc(DescriptorSetLayout&& setLayout)
    : setLayouts{std::move(setLayout)} {}

    PipelineLayoutDesc(DescriptorSetLayoutDesc&& setLayoutDesc)
    : setLayouts{std::move(setLayoutDesc)} {}
    
    PipelineLayoutDesc(PushConstantRange&& range)
    : pushConstants{std::move(range)} {}

    PipelineLayoutDesc&& operator|(SHADER_STAGE stages) &&
    {
        for (auto& setLayout : setLayouts) 
        {
            if (auto* setLayoutDesc = std::get_if<DescriptorSetLayoutDesc>(&setLayout)) 
                std::move(*setLayoutDesc) | stages;
            else  
            {
            /* 
            TODO: If the already created DescriptorSetLayout does not include the stages,
            we delete it and create a new DescriptorSetLayout with the given stages.
            */ 
            }
        }
        for (auto& range : pushConstants)
            std::move(range) | stages;
        return std::move(*this);
    }
};


inline PipelineLayoutDesc&& operator,(PipelineLayoutDesc&& desc, DescriptorSetLayoutInfo&& setLayout)
{
    desc.setLayouts.push_back(std::move(setLayout));
    return std::move(desc);
}

inline PipelineLayoutDesc&& operator,(PipelineLayoutDesc&& desc, PushConstantRange&& range)
{
    desc.pushConstants.push_back(std::move(range));
    return std::move(desc);
}




/**/
constexpr inline VkDescriptorPoolSize operator<=(
    VkDescriptorType type, int count)
{
    return {
        .type = type,
        .descriptorCount = (uint32_t)count,
    };
}


std::string operator""_file2str(const char* filename, size_t);



struct ComputePipelineCreateInfo {
    const char* csSrc;
    std::optional<PipelineLayout> layout;
};

struct BufferCreateInfo {
    uint64_t size;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags reqMemProps;
};

struct DescriptorPoolCreateInfo {
    std::vector<VkDescriptorPoolSize> maxTypes;
    uint32_t maxSets;
};

struct QueueSelector {
    const Device device;
    const uint32_t index;

    QueueSelector(Device device, uint32_t index) 
    : device(device), index(index) {} 

    Queue operator()(CommandBuffer cmdBuffer) const
    {
        return device.queue(cmdBuffer.type(), index);
    }

    Queue submit(CommandBuffer cmdBuffer) const
    {
        return (*this)(cmdBuffer).submit(cmdBuffer);
    }

    Queue submit(std::vector<CommandBuffer> cmdBuffers) const
    {
        return (*this)(cmdBuffers[0]).submit(std::move(cmdBuffers));
    }

    Queue submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence = std::nullopt) const
    {
        return (*this)(std::get<1>(batches[0])[0]).submit(std::move(batches), fence);
    }
};


struct PIPELINE_STAGE {
    struct T {
        uint64_t flag;
        T(uint64_t flag) : flag(flag) {}
        PIPELINE_STAGE operator|(T other) const { 
            return {flag | other.flag}; 
        }
    } flags;

    PIPELINE_STAGE(T flag) : flags(flag) {}

    PIPELINE_STAGE operator|(PIPELINE_STAGE other) const { 
        return {flags | other.flags}; 
    }
    
    operator uint64_t() const { 
        return flags.flag; 
    }

    operator uint32_t() const { 
        return (uint32_t)flags.flag; 
    }

    inline static const T NONE                              =              0ULL;
    inline static const T TOP_OF_PIPE                       =     0x00000001ULL;
    inline static const T DRAW_INDIRECT                     =     0x00000002ULL;
    inline static const T VERTEX_INPUT                      =     0x00000004ULL;
    inline static const T VERTEX_SHADER                     =     0x00000008ULL;
    inline static const T TESSELLATION_CONTROL_SHADER       =     0x00000010ULL;
    inline static const T TESSELLATION_EVALUATION_SHADER    =     0x00000020ULL;
    inline static const T GEOMETRY_SHADER                   =     0x00000040ULL;
    inline static const T FRAGMENT_SHADER                   =     0x00000080ULL;
    inline static const T EARLY_FRAGMENT_TESTS              =     0x00000100ULL;
    inline static const T LATE_FRAGMENT_TESTS               =     0x00000200ULL;
    inline static const T COLOR_ATTACHMENT_OUTPUT           =     0x00000400ULL;
    inline static const T COMPUTE_SHADER                    =     0x00000800ULL;
    inline static const T TRANSFER                          =     0x00001000ULL;
    inline static const T BOTTOM_OF_PIPE                    =     0x00002000ULL;
    inline static const T HOST                              =     0x00004000ULL;
    inline static const T ALL_GRAPHICS                      =     0x00008000ULL;
    inline static const T ALL_COMMANDS                      =     0x00010000ULL;
    inline static const T COMMAND_PREPROCESS                =     0x00020000ULL;
    inline static const T CONDITIONAL_RENDERING             =     0x00040000ULL;
    inline static const T TASK_SHADER                       =     0x00080000ULL;
    inline static const T MESH_SHADER                       =     0x00100000ULL;
    inline static const T RAY_TRACING_SHADER                =     0x00200000ULL;
    inline static const T FRAGMENT_SHADING_RATE_ATTACHMENT  =     0x00400000ULL;
    inline static const T FRAGMENT_DENSITY_PROCESS          =     0x00800000ULL;
    inline static const T TRANSFORM_FEEDBACK                =     0x01000000ULL;
    inline static const T ACCELERATION_STRUCTURE_BUILD      =     0x02000000ULL;
#ifdef VULKAN_VERSION_1_3
    inline static const T VIDEO_DECODE                      =     0x04000000ULL;
    inline static const T VIDEO_ENCODE                      =     0x08000000ULL;
    inline static const T ACCELERATION_STRUCTURE_COPY       =     0x10000000ULL;
    inline static const T OPTICAL_FLOW                      =     0x20000000ULL;
    inline static const T MICROMAP_BUILD                    =     0x40000000ULL;
    inline static const T COPY                              =    0x100000000ULL;
    inline static const T RESOLVE                           =    0x200000000ULL;
    inline static const T BLIT                              =    0x400000000ULL;
    inline static const T CLEAR                             =    0x800000000ULL;
    inline static const T INDEX_INPUT                       =   0x1000000000ULL;
    inline static const T VERTEX_ATTRIBUTE_INPUT            =   0x2000000000ULL;
    inline static const T PRE_RASTERIZATION_SHADERS         =   0x4000000000ULL;
    inline static const T CONVERT_COOPERATIVE_VECTOR_MATRIX = 0x100000000000ULL;
#endif
};


struct ACCESS {
    struct T {
        uint64_t flag;
        T(uint64_t flag) : flag(flag) {}
        ACCESS operator|(T other) const { 
            return {flag | other.flag}; 
        }  
    } flags;

    ACCESS(T flag) : flags(flag) {}

    ACCESS operator|(ACCESS other) const { 
        return {flags | other.flags}; 
    }
    
    operator uint64_t() const { 
        return flags.flag; 
    }

    operator uint32_t() const { 
        return (uint32_t)flags.flag; 
    }
    
    inline static const T NONE                                  =              0ULL;
    inline static const T INDIRECT_COMMAND_READ                 =     0x00000001ULL;
    inline static const T INDEX_READ                            =     0x00000002ULL;
    inline static const T VERTEX_ATTRIBUTE_READ                 =     0x00000004ULL;
    inline static const T UNIFORM_READ                          =     0x00000008ULL;
    inline static const T INPUT_ATTACHMENT_READ                 =     0x00000010ULL;
    inline static const T SHADER_READ                           =     0x00000020ULL;
    inline static const T SHADER_WRITE                          =     0x00000040ULL;
    inline static const T COLOR_ATTACHMENT_READ                 =     0x00000080ULL;
    inline static const T COLOR_ATTACHMENT_WRITE                =     0x00000100ULL;
    inline static const T DEPTH_STENCIL_ATTACHMENT_READ         =     0x00000200ULL;
    inline static const T DEPTH_STENCIL_ATTACHMENT_WRITE        =     0x00000400ULL;
    inline static const T TRANSFER_READ                         =     0x00000800ULL;
    inline static const T TRANSFER_WRITE                        =     0x00001000ULL;
    inline static const T HOST_READ                             =     0x00002000ULL;
    inline static const T HOST_WRITE                            =     0x00004000ULL;
    inline static const T MEMORY_READ                           =     0x00008000ULL;
    inline static const T MEMORY_WRITE                          =     0x00010000ULL;
    inline static const T COMMAND_PREPROCESS_READ               =     0x00020000ULL;
    inline static const T COMMAND_PREPROCESS_WRITE              =     0x00040000ULL;
    inline static const T COLOR_ATTACHMENT_READ_NONCOHERENT     =     0x00080000ULL;
    inline static const T CONDITIONAL_RENDERING_READ            =     0x00100000ULL;
    inline static const T ACCELERATION_STRUCTURE_READ           =     0x00200000ULL;
    inline static const T ACCELERATION_STRUCTURE_WRITE          =     0x00400000ULL;
    inline static const T FRAGMENT_SHADING_RATE_ATTACHMENT_READ =     0x00800000ULL;
    inline static const T FRAGMENT_DENSITY_MAP_READ             =     0x01000000ULL;
    inline static const T TRANSFORM_FEEDBACK_WRITE              =     0x02000000ULL;
    inline static const T TRANSFORM_FEEDBACK_COUNTER_READ       =     0x04000000ULL;
    inline static const T TRANSFORM_FEEDBACK_COUNTER_WRITE      =     0x08000000ULL;
#ifdef VULKAN_VERSION_1_3
    inline static const T SHADER_SAMPLED_READ                   =    0x100000000ULL;
    inline static const T SHADER_STORAGE_READ                   =    0x200000000ULL;
    inline static const T SHADER_STORAGE_WRITE                  =    0x400000000ULL;
    inline static const T VIDEO_DECODE_READ                     =    0x800000000ULL;
    inline static const T VIDEO_DECODE_WRITE                    =   0x1000000000ULL;
    inline static const T VIDEO_ENCODE_READ                     =   0x2000000000ULL;
    inline static const T VIDEO_ENCODE_WRITE                    =   0x4000000000ULL;
    inline static const T SHADER_BINDING_TABLE_READ             =  0x10000000000ULL;
    inline static const T DESCRIPTOR_BUFFER_READ                =  0x20000000000ULL;
    inline static const T OPTICAL_FLOW_READ                     =  0x40000000000ULL;
    inline static const T OPTICAL_FLOW_WRITE                    =  0x80000000000ULL;
    inline static const T MICROMAP_READ                         = 0x100000000000ULL;
    inline static const T MICROMAP_WRITE                        = 0x200000000000ULL;
#endif
};


struct PIPELINE_STAGE_ACCESS {
    PIPELINE_STAGE stage;
    ACCESS access = ACCESS::NONE;

    PIPELINE_STAGE_ACCESS(PIPELINE_STAGE::T stage)
    : stage(stage) {}

    PIPELINE_STAGE_ACCESS(PIPELINE_STAGE stage, ACCESS access=ACCESS::NONE)
    : stage(stage), access(access) {}

    PIPELINE_STAGE_ACCESS operator|(PIPELINE_STAGE_ACCESS other) const
    {
        return {stage | other.stage, access | other.access};
    }
};

inline PIPELINE_STAGE_ACCESS operator,(PIPELINE_STAGE stage, ACCESS access)
{
    return {stage, access};
}

struct MemoryBarrier {
    PIPELINE_STAGE_ACCESS srcMask = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    PIPELINE_STAGE_ACCESS dstMask = {PIPELINE_STAGE::NONE, ACCESS::NONE};
};

inline MemoryBarrier operator/(PIPELINE_STAGE_ACCESS mask1, PIPELINE_STAGE_ACCESS mask2)
{
    return {mask1, mask2};
}

struct BufferMemoryBarrier {
    PIPELINE_STAGE_ACCESS srcMask = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    PIPELINE_STAGE_ACCESS dstMask = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    Buffer buffer;
    uint64_t offset = 0;
    uint64_t size = VK_WHOLE_SIZE;
};

inline BufferMemoryBarrier operator/(PIPELINE_STAGE_ACCESS mask, Buffer buffer)
{
    return {
        .srcMask = mask,
        .buffer = buffer,
    };
}

inline BufferMemoryBarrier operator/(QueueType queueType, Buffer buffer)
{
    return {
        .opType = OwnershipTransferOpType::acquire,
        .pairedQueue = queueType,
        .buffer = buffer,
    };
}

inline BufferMemoryBarrier operator/(Buffer buffer, PIPELINE_STAGE_ACCESS mask)
{
    return {
        .dstMask = mask,
        .buffer = buffer,
    };
}

inline BufferMemoryBarrier operator/(Buffer buffer, QueueType queueType)
{
    return {
        .opType = OwnershipTransferOpType::release,
        .pairedQueue = queueType,
        .buffer = buffer,
    };
}

inline BufferMemoryBarrier&& operator/(PIPELINE_STAGE_ACCESS mask, BufferMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(QueueType queueType, BufferMemoryBarrier&& barrier)
{
    barrier.opType = OwnershipTransferOpType::acquire;
    barrier.pairedQueue = queueType;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(BufferMemoryBarrier&& barrier, PIPELINE_STAGE_ACCESS mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(BufferMemoryBarrier&& barrier, QueueType queueType)
{
    barrier.opType = OwnershipTransferOpType::release;
    barrier.pairedQueue = queueType;
    return std::move(barrier);
}

struct ImageMemoryBarrier {};

/*
버퍼 range class가 꼭 필요한가?
버퍼 range 필요 시점:
- vkMapMemory
- vkFlushMappedMemoryRanges, vkInvalidateMappedMemoryRanges
- vkCmdCopyBuffer, vkCmdUpdateBuffer, vkCmdFillBuffer 
- VkDescriptorBufferInfo 
- VkBufferMemoryBarrier 
- VkBufferViewCreateInfo 
*/
struct BufferRange {
    const Buffer buffer;
    const uint64_t offset;
    const uint64_t size;

    BufferRange() = delete;

    BufferRange(
        Buffer buffer, 
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE)
    : buffer(buffer)
    , offset(offset)
    , size(size) {}

    void flush() const
    {
        buffer.flush(offset, size);
    }

    void invalidate() const
    {
        buffer.invalidate(offset, size);
    }

    VkDescriptorBufferInfo descInfo() const
    {
        return buffer.descInfo(offset, size);
    }

    BufferMemoryBarrier barrier(
        PIPELINE_STAGE_ACCESS srcMask,
        PIPELINE_STAGE_ACCESS dstMask,
        OwnershipTransferOpType opType=OwnershipTransferOpType::none,
        QueueType queueType=queue_max
    ) const
    {
        return buffer.barrier(
            srcMask, dstMask, 
            opType, queueType, 
            offset, size
        );
    }
};



struct SemaphoreStage {
    const Semaphore sem;
    const PIPELINE_STAGE stage;

    SemaphoreStage(
        Semaphore sem, 
        PIPELINE_STAGE stage=PIPELINE_STAGE::ALL_COMMANDS) 
    : sem(sem), stage(stage) {}
};

inline SemaphoreStage Semaphore::operator[](PIPELINE_STAGE stage) const
{
    return {*this, stage};
}


inline std::vector<SemaphoreStage> operator,(SemaphoreStage sem1, SemaphoreStage sem2)
{
    return {sem1, sem2};
}   

inline std::vector<SemaphoreStage>&& operator,(std::vector<SemaphoreStage>&& sems, SemaphoreStage sem)
{
    sems.push_back(sem);
    return std::move(sems);
}

inline std::vector<CommandBuffer> operator,(CommandBuffer cmdBuffer1, CommandBuffer cmdBuffer2)
{
    return {cmdBuffer1, cmdBuffer2};
}

inline std::vector<CommandBuffer>&& operator,(std::vector<CommandBuffer>&& cmdBuffers, CommandBuffer cmdBuffer)
{
    cmdBuffers.push_back(cmdBuffer);
    return std::move(cmdBuffers);
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, CommandBuffer cmdBuffer)
{
    return {{sem}, {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, CommandBuffer cmdBuffer)
{
    return {std::move(sems), {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, SemaphoreStage sem)
{
    return {{}, {cmdBuffer}, {sem}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, std::vector<SemaphoreStage>&& sems)
{
    return {{}, {cmdBuffer}, std::move(sems)};
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {{sem}, std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {std::move(sems), std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, SemaphoreStage sem)
{
    return {{}, std::move(cmdBuffers), {sem}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, std::vector<SemaphoreStage>&& sems)
{
    return {{}, std::move(cmdBuffers), std::move(sems)};
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, SemaphoreStage sem)
{
    std::get<2>(batch).push_back(sem);
    return std::move(batch);
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, std::vector<SemaphoreStage>&& sems)
{
    std::get<2>(batch) = std::move(sems);
    return std::move(batch);
}

struct Waiting {};
inline void waiting(Waiting w){}

struct Submitting {
private:
    friend Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch);
    friend Submitting operator<<(Queue queue, CommandBuffer cmdBuffer);
    friend Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers);
    friend Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch);
    friend Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer);
    friend Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers);
    friend void operator<<(Submitting&& submitting, Fence fence);
    friend void operator<<(Submitting&& submitting, void(Waiting));

    Submitting() = delete;
    Submitting(const Submitting&) = delete;
    Submitting(Submitting&&) = delete;

    Submitting(Queue queue, SubmissionBatchInfo&& batch) : queue(queue)
    {
        batches.emplace_back(std::move(batch));
    }
    Queue queue;
    std::vector<SubmissionBatchInfo> batches;
    bool isWaiting = false;
    std::optional<Fence> fence;
    
public:
    ~Submitting() { 
        queue.submit(std::move(batches), fence); 
       
        if (isWaiting) {
            queue.waitIdle();
        }
    }
};


inline Submitting operator<<(Queue queue, CommandBuffer cmdBuffer)
{
    return Submitting(
        queue,
        std::make_tuple(
            std::vector<vk::SemaphoreStage>{},          // no wait semaphores
            std::vector<CommandBuffer>{cmdBuffer},      // single command
            std::vector<vk::SemaphoreStage>{}           // no signal semaphores
        )
    );
}

inline Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers)
{
    return Submitting(
        queue,
        std::make_tuple(
            std::vector<vk::SemaphoreStage>{},
            std::move(cmdBuffers),
            std::vector<vk::SemaphoreStage>{}
        )
    );
}

inline Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch)
{
    return Submitting(queue, std::move(batch));
}

inline Submitting operator<<(QueueSelector queueSelector, CommandBuffer cmdBuffer)
{
    return operator<<(queueSelector(cmdBuffer), cmdBuffer);
}

inline Submitting operator<<(QueueSelector queueSelector, std::vector<CommandBuffer>&& cmdBuffers)
{
    return operator<<(queueSelector(cmdBuffers[0]), std::move(cmdBuffers));
}

inline Submitting operator<<(QueueSelector queueSelector, SubmissionBatchInfo&& batch)
{
    return operator<<(queueSelector(std::get<1>(batch)[0]), std::move(batch));
}

inline Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::vector<CommandBuffer>{cmdBuffer}, 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::move(cmdBuffers), 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch)
{   
    submitting.batches.emplace_back(std::move(batch));
    return std::move(submitting);
}

inline void operator<<(Submitting&& submitting, Fence fence)
{
    submitting.fence = fence;
}

inline void operator<<(Submitting&& submitting, void(Waiting))
{
    submitting.isWaiting = true;
}





} // namespace vk

#endif // VULKAN_APP_H