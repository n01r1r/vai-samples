#include <array>
#include <vector>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include <vulkan/vulkan_core.h>
#include <spirv_reflect.h>
#include "templateHelper.h"
#include "vulkanApp.h"

using namespace vk;

#define GLSLANG_STAGE_MAPPING(vk, glslang) case vk: glslang_stage = glslang; break

std::vector<uint32_t> glsl2spv(VkShaderStageFlags stage, const char* shaderSource) 
{
    glslang_stage_t glslang_stage;

    switch (stage) 
    {
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_VERTEX_BIT, GLSLANG_STAGE_VERTEX);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_FRAGMENT_BIT, GLSLANG_STAGE_FRAGMENT);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_COMPUTE_BIT, GLSLANG_STAGE_COMPUTE);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_RAYGEN_BIT_KHR, GLSLANG_STAGE_RAYGEN);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, GLSLANG_STAGE_ANYHIT);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, GLSLANG_STAGE_CLOSESTHIT);
    GLSLANG_STAGE_MAPPING(VK_SHADER_STAGE_MISS_BIT_KHR, GLSLANG_STAGE_MISS);
    default: throw;
    }

    const glslang_input_t input = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = glslang_stage,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_3,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_5,
        .code = shaderSource,
        .default_version = 100,
        .default_profile = GLSLANG_NO_PROFILE,
        .force_default_version_and_profile = false,
        .forward_compatible = false,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
        .resource = glslang_default_resource(),
    };

    glslang_shader_t* shader = glslang_shader_create(&input);

    if (!glslang_shader_preprocess(shader, &input)) {
        printf("GLSL preprocessing failed (%d)\n", stage);
        printf("%s\n", glslang_shader_get_info_log(shader));
        printf("%s\n", glslang_shader_get_info_debug_log(shader));
        printf("%s\n", input.code);
        glslang_shader_delete(shader);
        return {};
    }

    if (!glslang_shader_parse(shader, &input)) {
        printf("GLSL parsing failed (%d)\n", stage);
        printf("%s\n", glslang_shader_get_info_log(shader));
        printf("%s\n", glslang_shader_get_info_debug_log(shader));
        printf("%s\n", glslang_shader_get_preprocessed_code(shader));
        glslang_shader_delete(shader);
        return {};
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        printf("GLSL linking failed (%d)\n", stage);
        printf("%s\n", glslang_program_get_info_log(program));
        printf("%s\n", glslang_program_get_info_debug_log(program));
        glslang_program_delete(program);
        glslang_shader_delete(shader);
        return {};
    }

    glslang_program_SPIRV_generate(program, glslang_stage);

    size_t size = glslang_program_SPIRV_get_size(program);
    std::vector<uint32_t> spvBirary(size);
    glslang_program_SPIRV_get(program, spvBirary.data());

    const char* spirv_messages = glslang_program_SPIRV_get_messages(program);
    if (spirv_messages)
        printf("(%d) %s\b", stage, spirv_messages);

    glslang_program_delete(program);
    glslang_shader_delete(shader);

    return spvBirary;
}


void* createReflectShaderModule(const std::vector<uint32_t>& spvBinary)
{
    SpvReflectShaderModule* pModule = new SpvReflectShaderModule;
    SpvReflectResult result = spvReflectCreateShaderModule(
        spvBinary.size() * sizeof(uint32_t),
        spvBinary.data(),
        pModule
    );

    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        printf("Failed to create SPIR-V Reflect Shader Module: %d\n", result);
        return {};
    }

    return pModule;
}

void destroyReflectShaderModule(void* pModule)
{
    spvReflectDestroyShaderModule((SpvReflectShaderModule*)pModule);
}

std::array<uint32_t, 3> extractWorkGroupSize(const void* pModule)
{
    const SpvReflectShaderModule& module = *(const SpvReflectShaderModule*)pModule;
    std::array<uint32_t, 3> workGroupSize = { 1, 1, 1 };

    if (module.spirv_execution_model == SpvExecutionModelGLCompute) 
    {
        _ASSERT(module.entry_point_count == 1);

        const auto& entryPoint = module.entry_points[0];
        workGroupSize[0] = entryPoint.local_size.x;
        workGroupSize[1] = entryPoint.local_size.y;
        workGroupSize[2] = entryPoint.local_size.z;
    }

    return workGroupSize;
}

PipelineLayoutDesc extractPipelineLayoutDesc(const void* pModule)
{
    PipelineLayoutDesc desc;
    const SpvReflectShaderModule& module = *(const SpvReflectShaderModule*)pModule;
    
    std::vector<SpvReflectDescriptorSet*> srcSetLayouts = arrayFrom(spvReflectEnumerateDescriptorSets, &module);
    desc.setLayouts.reserve(srcSetLayouts.size());

    for (const auto* pSrcSetLayout : srcSetLayouts) 
    {
        const SpvReflectDescriptorSet& srcSetLayout = *pSrcSetLayout;
        const uint32_t numBindings = srcSetLayout.binding_count;

        DescriptorSetLayoutDesc dstSrcLayout;
        dstSrcLayout.bindings.reserve(numBindings);

        for (uint32_t j = 0; j < numBindings; ++j) 
        {
            const SpvReflectDescriptorBinding& srcBinding = *srcSetLayout.bindings[j];

            dstSrcLayout.bindings.emplace_back(
                srcBinding.binding,
                srcBinding.descriptor_type,
                [&] {
                    uint32_t count = 1;
                    for (uint32_t k = 0; k < srcBinding.array.dims_count; ++k)
                        count *= srcBinding.array.dims[k];
                    return count;
                }(),
                module.shader_stage
            );
        }

        desc.setLayouts.push_back(std::move(dstSrcLayout));
    }

    SpvReflectResult result;
    const SpvReflectBlockVariable* pcBlock = spvReflectGetEntryPointPushConstantBlock(&module, "main", &result);
    if (result == SPV_REFLECT_RESULT_SUCCESS && pcBlock) 
    {
        desc.pushConstants.push_back(
            PushConstantRange(
                pcBlock->offset, 
                pcBlock->size
            ) | module.shader_stage
        );
    }

    return desc;
}

