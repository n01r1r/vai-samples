cmake_minimum_required(VERSION 3.16)

project(110-GPT2-hyungkyu)

add_executable(${PROJECT_NAME}
    main.cpp
    tokenizer/test.cpp
    tokenizer/bpeTokenizer.cpp
    dataloader/test.cpp
    dataloader/gptDataset.cpp
    dataloader/gptDataLoader.cpp
    model/embedding/embedding.cpp
    core/vulkanApp.cpp
    core/spirvHelpers.cpp
    core/jsonParser.cpp
    model/embedding/embeddingNode.cpp
    model/embedding/embeddingNodeTest.cpp
    model/attention/attentionNode.cpp
    model/attention/attentionNodeTest.cpp
    model/transformer/transformerNode.cpp
    model/transformer/transformerNodeTest.cpp
    model/gpt2.cpp
    model/gpt2Test.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    nlohmann_json
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
)

