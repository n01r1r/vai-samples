#ifndef ERROR_H
#define ERROR_H

#include <vulkan/vulkan_core.h>
#include <source_location>
#include <cstdlib>
#include <cstdio>


template<typename T>
inline void assert_impl(
    const T& condition,
    const char* condition_str,
    const std::source_location location = std::source_location::current()
) {
#ifndef NDEBUG
    if (!static_cast<bool>(condition)) {
        fprintf(stderr,
            "[ASSERT FAILED] \"%s\" at %s in %s(%d:%d)\n",
            condition_str,
            location.function_name(),
            location.file_name(),
            static_cast<int>(location.line()),
            static_cast<int>(location.column())
        );
        std::abort();
    }
#endif
}

#define ASSERT_(expr) assert_impl((expr), #expr, std::source_location::current())
#define _ASSERT(expr) ASSERT_(expr)  // Alias for compatibility



inline const char* vkResult2String(VkResult errorCode)
{
    switch (errorCode)
    {
#define STR(r) case VK_ ##r: return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
        STR(ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
#undef STR
    default:
        return "UNKNOWN_ERROR";
    }
}

inline void operator!(VkResult vr)
{
    if (vr != VK_SUCCESS)
    {
        fprintf(stderr, "Fatal : VkResult is \"%s\"\n", vkResult2String(vr));
		throw vr;
    }
}


#endif // ERROR_