#include <GLFW/glfw3.h>
#include "neuralNodes.h"
#include "efficientNet.h"

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 800;

GLFWwindow* createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    return glfwCreateWindow(WIDTH, HEIGHT, "EfficientNet", nullptr, nullptr);
}

int main()
{
    glfwInit();
    // GLFWwindow* window = createWindow();
    
    void loadShaders();
    loadShaders();
    
    // TODO: Initialize Vulkan device
    // TODO: Create EfficientNet network
    // TODO: Load weights from JSON
    // TODO: Load and preprocess input image
    // TODO: Run inference
    // TODO: Process and display results
    
    // while (!glfwWindowShouldClose(window))
    // {
    //     glfwPollEvents();
    // }
    
    // glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
