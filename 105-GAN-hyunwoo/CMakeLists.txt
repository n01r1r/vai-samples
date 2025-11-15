cmake_minimum_required(VERSION 3.16)

project(105-GAN-hyunwoo)

set(CMAKE_CXX_STANDARD 23)

# Set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Define parent directory for accessing external libraries
set(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include parent's external directory
include_directories(${PARENT_DIR}/external)

# Add project definitions
add_definitions(
    -DPROJECT_CURRENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -DPROJECT_ROOT_DIR="${PARENT_DIR}"
)

# Add external libraries from parent project
set(SPIRV-Headers_SOURCE_DIR "${PARENT_DIR}/external/spirv-headers")
set(mimalloc_SOURCE_DIR "${PARENT_DIR}/external/mimalloc")

################################################################################
# SPIRV-Tools
################################################################################
message(STATUS "[SPIRV-Tools] Configuring SPIRV-Tools...")
add_subdirectory(${PARENT_DIR}/external/SPIRV-Tools ${CMAKE_BINARY_DIR}/SPIRV-Tools EXCLUDE_FROM_ALL)
message(STATUS "[SPIRV-Tools] SPIRV-Tools configuration completed.")

################################################################################
# SPIRV-Reflect
################################################################################
message(STATUS "[SPIRV-Reflect] Configuring SPIRV-Reflect...")
set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "Build spirv-reflect executable")
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "Build a SPIRV-Reflect static library")
add_subdirectory(${PARENT_DIR}/external/SPIRV-Reflect ${CMAKE_BINARY_DIR}/SPIRV-Reflect EXCLUDE_FROM_ALL)
message(STATUS "[SPIRV-Reflect] SPIRV-Reflect configuration completed.")

################################################################################
# glslang
################################################################################
message(STATUS "[glslang] Configuring glslang...")
set(BUILD_EXTERNAL OFF CACHE BOOL "Disable building external dependencies")
set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "Allow external SPIRV-Tools")
set(ENABLE_HLSL OFF CACHE BOOL "Disable HLSL support")
set(ENABLE_RTTI ON CACHE BOOL "Enable RTTI for glslang")
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "Disable glslang binaries")
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "Skip installation of SPIRV-Tools")
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "Skip building SPIRV-Tools executables")
add_subdirectory(${PARENT_DIR}/external/glslang ${CMAKE_BINARY_DIR}/glslang EXCLUDE_FROM_ALL)
message(STATUS "[glslang] glslang configuration completed.")

################################################################################
# GLFW
################################################################################
message(STATUS "[GLFW] Configuring GLFW...")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation")
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target")
add_subdirectory(${PARENT_DIR}/external/glfw ${CMAKE_BINARY_DIR}/glfw EXCLUDE_FROM_ALL)
message(STATUS "[GLFW] GLFW configuration completed.")

################################################################################
# JSON
################################################################################
message(STATUS "[json] Configuring json...")
add_subdirectory(${PARENT_DIR}/external/json ${CMAKE_BINARY_DIR}/json EXCLUDE_FROM_ALL)
message(STATUS "[json] json configuration completed.")

################################################################################
# Main executable
################################################################################
add_executable(${PROJECT_NAME}
    main.cpp
    test.cpp
    benchmark.cpp
    vulkanApp.cpp
    neuralNodes.cpp
    spirvHelpers.cpp
    jsonParser.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLFW_INCLUDE_VULKAN
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    spirv-reflect-static
    ${Vulkan_LIBRARIES}
    glslang
    glslang-default-resource-limits
    nlohmann_json
)
