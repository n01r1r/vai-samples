#include <GLFW/glfw3.h>

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 800;

GLFWwindow* createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan App", nullptr, nullptr);
}

int main()
{
    glfwInit();
    //GLFWwindow* window = createWindow();
    
    void test();
    test();

    // while (!glfwWindowShouldClose(window))
    // {
    //     glfwPollEvents();
    // }

    //glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}