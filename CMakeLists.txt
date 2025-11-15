cmake_minimum_required(VERSION 3.16)

project(vulkan-raytracing-lectures)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_definitions(
    -DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

include_directories(
    ${CMAKE_SOURCE_DIR}/external
)

option(REQUIRE_BACKEND_VULKAN "Require Vulkan backend" ON)
option(REQUIRE_BACKEND_OPENCL "Require OpenCL backend" OFF)
option(REQUIRE_BACKEND_CUDA "Require CUDA backend" OFF)
option(USE_KOMPUTE "Use Kompute library" OFF)
option(USE_CTRE "Use ctre library" OFF)


################################################################################
# OpenCL
################################################################################
if(REQUIRE_BACKEND_OPENCL)
    message(STATUS "[OpenCL] Configuring OpenCL SDK...")
    set(BUILD_TESTING OFF CACHE BOOL "Disable building tests for OpenCL SDK")
    set(BUILD_DOCS OFF CACHE BOOL "Disable building documentation for OpenCL SDK")
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable building examples for OpenCL SDK")
    set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "Disable building OpenCL SDK samples")
    add_subdirectory(external/OpenCL-SDK)
    message(STATUS "[OpenCL] OpenCL SDK configuration completed.")
endif()


################################################################################
# CUDA
################################################################################
if (REQUIRE_BACKEND_CUDA)
    message(STATUS "[CUDA] Configuring CUDA Toolkit...")

    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA Toolkit not found. Please install the CUDA Toolkit.")
    endif()
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    message(STATUS "[CUDA] CUDA Toolkit configuration completed.")
endif()


################################################################################
# Vulkan
################################################################################
if (REQUIRE_BACKEND_VULKAN)
    message(STATUS "[Vulkan] Configuring Vulkan SDK...")

    find_package(Vulkan)
    if (NOT Vulkan_FOUND)
        message(FATAL_ERROR "Vulkan not found. Please install the Vulkan SDK.")
    endif()

    message(STATUS "[Vulkan] Vulkan SDK configuration completed.")
endif()


################################################################################
# SPIRV-Tools
################################################################################
if (Vulkan_FOUND)
    message(STATUS "[SPIRV-Tools] Configuring SPIRV-Tools...")
    set(SPIRV-Headers_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/spirv-headers")
    set(mimalloc_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/mimalloc")
    add_subdirectory(external/SPIRV-Tools)
    message(STATUS "[SPIRV-Tools] SPIRV-Tools configuration completed.")
endif()


################################################################################
# SPIRV-Reflect
################################################################################
if (Vulkan_FOUND)
    message(STATUS "[SPIRV-Reflect] Configuring SPIRV-Reflect...")
    set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "Build spirv-reflect executable")
    set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "Build a SPIRV-Reflect static library")
    # set(SPIRV_REFLECT_EXAMPLES OFF CACHE BOOL "Build the SPIRV-Reflect example programs")
    # set(SPIRV_REFLECT_ENABLE_ASSERTS ON CACHE BOOL "Enable asserts for debugging")
    add_subdirectory(external/SPIRV-Reflect)
    message(STATUS "[SPIRV-Reflect] SPIRV-Reflect configuration completed.")
endif()


################################################################################
# glslang
################################################################################
if (Vulkan_FOUND)
    message(STATUS "[glslang] Configuring glslang...")
    # set(ENABLE_OPT OFF CACHE BOOL "Disable glslang optimizer, which requires SPIRV-Tools")
    set(BUILD_EXTERNAL OFF CACHE BOOL "Disable building external dependencies")
    set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "Allow external SPIRV-Tools")
    set(ENABLE_HLSL OFF CACHE BOOL "Disable HLSL support")
    set(ENABLE_RTTI ON CACHE BOOL "Enable RTTI for glslang, off invokes a lot of warnings")
    
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "Disable glslang binaries")
    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "Skip installation of SPIRV-Tools")
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "Skip building SPIRV-Tools executables")
    # set(ENABLE_RTTI ON CACHE BOOL "Enable RTTI for glslang, off invokes a lot of warnings")
    add_subdirectory(external/glslang)
    
    message(STATUS "[glslang] glslang configuration completed.")
endif()


################################################################################
# GLFW
################################################################################
if (Vulkan_FOUND)
    message(STATUS "[GLFW] Configuring GLFW...")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation")
    # set(BUILD_SHARED_LIBS ON CACHE BOOL "Build GLFW as shared library") # Uncomment if you want to build GLFW as .dll
    set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target") # default is ON, but I don't know when it needs
    add_subdirectory(external/glfw)
    add_definitions(-DGLFW_INCLUDE_VULKAN)
    message(STATUS "[GLFW] GLFW configuration completed.")
endif()


################################################################################
# ctre
################################################################################
if (USE_CTRE)
    message(STATUS "[ctre] Configuring ctre...")
    add_subdirectory(external/ctre)
    message(STATUS "[ctre] ctre configuration completed.")
endif()


################################################################################
# kompute
################################################################################
if (Vulkan_FOUND AND USE_KOMPUTE)
    message(STATUS "[Kompute] Configuring Kompute...")
    add_subdirectory(external/kompute)
    message(STATUS "[Kompute] Kompute configuration completed.")
endif()


################################################################################
# json
################################################################################
message(STATUS "[json] Configuring json...")
add_subdirectory(external/json)
message(STATUS "[json] json configuration completed.")


################################################################################
# main-project
################################################################################

add_subdirectory(10-mnist)
add_subdirectory(11-mnist-refactor)
# add_subdirectory(114-MobileNetV2-hwPark)
