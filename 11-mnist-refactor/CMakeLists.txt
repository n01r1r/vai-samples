cmake_minimum_required(VERSION 3.16)

project(11-mnist-refactor)

add_executable(${PROJECT_NAME} 
    main.cpp
    test.cpp
    vulkanApp.cpp
    neuralNodes.cpp
    spirvHelpers.cpp
    stb_image.cpp
    jsonParser.cpp
)

add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    nlohmann_json
)