#define VULKAN_HPP_NO_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/hash.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "array"
#include "cstdlib"
#include "cstring"
#include "exception"
#include "fstream"
#include "iostream"
#include "map"
#include "optional"
#include "random"
#include "set"
#include "stdexcept"
#include "unordered_map"
#include "vector"

#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = (VkResult)x;                                                \
    if (err) {                                                                 \
      std::cout << "Detected Vulkan error: " << err << std::endl;              \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static void check_vk_result(VkResult err) {
  if (err == 0)
    return;
  fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
  if (err < 0)
    abort();
}

const int MAX_FRAMES_IN_FLIGHT = 2;

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const uint32_t PARTICLE_COUNT = 8192;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    vk::KHRSwapchainExtensionName, vk::KHRSynchronization2ExtensionName};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

vk::Result CreateDebugUtilsMessengerEXT(
    vk::Instance instance,
    const vk::DebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const vk::AllocationCallbacks *pAllocator,
    vk::DebugUtilsMessengerEXT *pDebugMessenger) {
  vk::detail::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);
  return instance.createDebugUtilsMessengerEXT(pCreateInfo, pAllocator,
                                               pDebugMessenger, dldi);
}

void DestroyDebugUtilsMessengerEXT(vk::Instance instance,
                                   vk::DebugUtilsMessengerEXT debugMessenger,
                                   const vk::AllocationCallbacks *pAllocator) {
  vk::detail::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);
  instance.destroyDebugUtilsMessengerEXT(debugMessenger, pAllocator, dldi);
}

struct ModelPushConstants {
  glm::vec4 data;
  glm::mat4 model;
};

struct Texture {
  vk::Image image;
  vk::DeviceMemory imageMemory;
  vk::ImageView imageView;
  vk::Sampler sampler;
  uint32_t mipLevels;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;
  uint32_t textureIndex;

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord && textureIndex == other.textureIndex;
  }

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex};

    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 4>
  getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = vk::Format::eR32Uint;
    attributeDescriptions[3].offset = offsetof(Vertex, textureIndex);

    return attributeDescriptions;
  }
};

struct Model {
  uint32_t verticesOffset;
  uint32_t indicesOffset;
  uint32_t textureIndex;
  float angleX = 0.0f;
  float angleY = 0.0f;
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

struct Particle {
  glm::vec3 position;
  glm::vec2 velocity;
  glm::vec4 color;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription{
        .binding = 0,
        .stride = sizeof(Particle),
        .inputRate = vk::VertexInputRate::eVertex};

    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[0].offset = offsetof(Particle, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[1].offset = offsetof(Particle, color);

    return attributeDescriptions;
  }
};

struct Camera {
  float fov = 45.0f;
  glm::vec3 eye = {2.0f, 2.0f, 2.0f};
  glm::vec3 center = {0.0f, 0.0f, 0.0f};
  glm::vec3 up = {0.0f, 0.0f, 1.0f};
  glm::vec3 position = {0.0f, 0.0f, 0.0f};
};

namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
} // namespace std

struct UniformBufferObject {
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class TengineApp {
public:
  void run() {
    initWindow();
    initVulkan();
    initImGui();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window;

  vk::Instance instance;
  vk::DebugUtilsMessengerEXT debugMessenger;
  vk::SurfaceKHR surface;

  vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
  vk::Device device;

  vk::Queue graphicsQueue;
  vk::Queue computeQueue;
  vk::Queue presentQueue;

  vk::SwapchainKHR swapChain;
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::ImageView> swapChainImageViews;

  vk::RenderPass renderPass;
  vk::DescriptorSetLayout descriptorSetLayout;

  vk::PipelineCache pipelineCache;

  vk::PipelineLayout particlesPipelineLayout;
  vk::Pipeline particlesPipeline;

  vk::PipelineLayout graphicsPipelineLayout;
  vk::Pipeline graphicsPipeline;

  std::vector<vk::Framebuffer> swapChainFramebuffers;

  vk::CommandPool commandPool;
  std::vector<vk::CommandBuffer> commandBuffers;

  std::vector<vk::Semaphore> imageAvailableSemaphores;
  std::vector<vk::Semaphore> renderFinishedSemaphores;
  std::vector<vk::Fence> inFlightFences;

  bool framebufferResized = false;

  uint32_t currentFrame = 0;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vk::Buffer vertexBuffer;
  vk::DeviceMemory vertexBufferMemory;
  vk::Buffer indexBuffer;
  vk::DeviceMemory indexBufferMemory;

  std::vector<vk::Buffer> uniformBuffers;
  std::vector<vk::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  vk::DescriptorPool imGuiDescriptorPool;

  vk::DescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;

  std::vector<Texture> textures;

  vk::Image depthImage;
  vk::DeviceMemory depthImageMemory;
  vk::ImageView depthImageView;

  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

  vk::Image colorImage;
  vk::DeviceMemory colorImageMemory;
  vk::ImageView colorImageView;

  std::vector<vk::Buffer> shaderStorageBuffers;
  std::vector<vk::DeviceMemory> shaderStorageBuffersMemory;

  std::vector<vk::DescriptorSet> computeDescriptorSets;
  vk::DescriptorSetLayout computeDescriptorSetLayout;
  vk::PipelineLayout computePipelineLayout;
  vk::Pipeline computePipeline;
  std::vector<vk::CommandBuffer> computeCommandBuffers;
  std::vector<vk::Fence> computeInFlightFences;
  std::vector<vk::Semaphore> computeFinishedSemaphores;
  vk::DescriptorPool computeDescriptorPool;

  Camera camera;

  std::vector<Model> models;

  std::vector<const char *> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions,
                                         glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }

  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    VK_CHECK(vk::enumerateInstanceLayerProperties(&layerCount, nullptr));

    std::vector<vk::LayerProperties> availableLayers(layerCount);
    VK_CHECK(vk::enumerateInstanceLayerProperties(&layerCount,
                                                  availableLayers.data()));

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  vk::SampleCountFlagBits getMaxUsableSampleCount() {
    auto physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts =
        physicalDeviceProperties.limits.framebufferColorSampleCounts &
        physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64)
      return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32)
      return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16)
      return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8)
      return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4)
      return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2)
      return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
  }

  void initWindow() {
    if (!glfwInit())
      throw std::runtime_error("failed to initialize glfw!");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Tengine", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = reinterpret_cast<TengineApp *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void populateDebugMessengerCreateInfo(
      vk::DebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {
        .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback};
  }

  void initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    init_info.QueueFamily = indices.graphicsAndComputeFamily.value();

    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = pipelineCache;
    init_info.DescriptorPool = imGuiDescriptorPool;
    init_info.RenderPass = renderPass;
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.MSAASamples = (VkSampleCountFlagBits)msaaSamples;
    init_info.CheckVkResultFn = check_vk_result;

    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();
    ImGui_ImplVulkan_DestroyFontsTexture();
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createComputeDescriptorSetLayout();
    createDescriptorSetLayout();
    createParticlePipeline();
    createGraphicsPipeline();
    createComputePipeline();
    createCommandPool();
    createShaderStorageBuffers();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createTextureImages();
    createTextureImageViews();
    createTextureSamplers();
    loadModels();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createImGuiDescriptorPool();
    createDescriptorPool();
    createComputeDescriptorPool();
    createDescriptorSets();
    createComputeDescriptorSets();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
  }

  void recordComputeCommandBuffer(vk::CommandBuffer commandBuffer) {
    vk::CommandBufferBeginInfo beginInfo{
        .sType = vk::StructureType::eCommandBufferBeginInfo};

    VK_CHECK(commandBuffer.begin(&beginInfo));

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                               computePipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, 1,
        &computeDescriptorSets[currentFrame], 0, nullptr);

    commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);

    commandBuffer.end();
  }

  void createComputeDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings;
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    layoutBindings[0].pImmutableSamplers = nullptr;
    layoutBindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    layoutBindings[2].pImmutableSamplers = nullptr;
    layoutBindings[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
        .bindingCount = 3,
        .pBindings = layoutBindings.data()};

    VK_CHECK(device.createDescriptorSetLayout(&layoutInfo, nullptr,
                                              &computeDescriptorSetLayout));
  }

  void createComputePipeline() {
    auto computeShaderCode = readFile("shaders/bin/compute.comp.spv");

    vk::ShaderModule computeShaderModule =
        createShaderModule(computeShaderCode);

    vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = computeShaderModule,
        .pName = "main"};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .setLayoutCount = 1,
        .pSetLayouts = &computeDescriptorSetLayout};

    VK_CHECK(device.createPipelineLayout(&pipelineLayoutInfo, nullptr,
                                         &computePipelineLayout));

    vk::ComputePipelineCreateInfo pipelineInfo{
        .sType = vk::StructureType::eComputePipelineCreateInfo,
        .stage = computeShaderStageInfo,
        .layout = computePipelineLayout};

    VK_CHECK(device.createComputePipelines(VK_NULL_HANDLE, 1, &pipelineInfo,
                                           nullptr, &computePipeline));

    device.destroyShaderModule(computeShaderModule);
  }

  void createShaderStorageBuffers() {
    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto &particle : particles) {
      float r = 0.25f * sqrt(rndDist(rndEngine));
      float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846;
      float x = r * cos(theta) * HEIGHT / WIDTH;
      float y = r * sin(theta);
      float z = rndDist(rndEngine);
      particle.position = glm::vec3(x, y, z);
      particle.velocity = glm::normalize(glm::vec3(x, y, z)) * 0.00025f;
      particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                                 rndDist(rndEngine), 0.5f);
    }

    vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    VK_CHECK(device.mapMemory(stagingBufferMemory, 0, bufferSize,
                              vk::MemoryMapFlags(0), &data));
    memcpy(data, particles.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBufferMemory);

    shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      createBuffer(bufferSize,
                   vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eVertexBuffer |
                       vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eDeviceLocal,
                   shaderStorageBuffers[i], shaderStorageBuffersMemory[i]);
      copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
    }

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void createComputeCommandBuffers() {
    computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    vk::CommandBufferAllocateInfo allocInfo{
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = (uint32_t)computeCommandBuffers.size()};

    VK_CHECK(device.allocateCommandBuffers(&allocInfo,
                                           computeCommandBuffers.data()));
  }

  void createComputeDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .sType = vk::StructureType::eDescriptorSetAllocateInfo,
        .descriptorPool = computeDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()};

    computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    VK_CHECK(device.allocateDescriptorSets(&allocInfo,
                                           computeDescriptorSets.data()));

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo uniformBufferInfo{
          .buffer = uniformBuffers[i],
          .offset = 0,
          .range = sizeof(UniformBufferObject)};

      std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};
      descriptorWrites[0].sType = vk::StructureType::eWriteDescriptorSet;
      descriptorWrites[0].dstSet = computeDescriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

      vk::DescriptorBufferInfo storageBufferInfoLastFrame{
          .buffer = shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT],
          .offset = 0,
          .range = sizeof(Particle) * PARTICLE_COUNT};

      descriptorWrites[1].sType = vk::StructureType::eWriteDescriptorSet;
      descriptorWrites[1].dstSet = computeDescriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

      vk::DescriptorBufferInfo storageBufferInfoCurrentFrame{
          .buffer = shaderStorageBuffers[i],
          .offset = 0,
          .range = sizeof(Particle) * PARTICLE_COUNT};

      descriptorWrites[2].sType = vk::StructureType::eWriteDescriptorSet;
      descriptorWrites[2].dstSet = computeDescriptorSets[i];
      descriptorWrites[2].dstBinding = 2;
      descriptorWrites[2].dstArrayElement = 0;
      descriptorWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
      descriptorWrites[2].descriptorCount = 1;
      descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

      device.updateDescriptorSets(
          static_cast<uint32_t>(descriptorWrites.size()),
          descriptorWrites.data(), 0, nullptr);
    }
  }

  void loadModels() {
    loadModel("models/viking_room.obj");
    loadModel("models/craneo.obj");

    uint32_t prevVerticesOffset = 0;
    uint32_t prevIndiciesOffset = 0;

    uint32_t i = 0;

    for (Model &model : models) {
      model.textureIndex = i;

      model.verticesOffset = prevVerticesOffset;
      prevVerticesOffset += model.vertices.size() * sizeof(Vertex);

      for (Vertex &vert : model.vertices) {
        vert.textureIndex = i;
      }

      vertices.insert(vertices.end(), model.vertices.begin(),
                      model.vertices.end());

      model.indicesOffset = prevIndiciesOffset;
      prevIndiciesOffset += model.indices.size();
      indices.insert(indices.end(), model.indices.begin(), model.indices.end());

      i += 1;
    }
  }

  void loadModel(std::string path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          path.c_str())) {
      throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};
    Model model;

    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        Vertex vertex{.pos = {attrib.vertices[3 * index.vertex_index + 0],
                              attrib.vertices[3 * index.vertex_index + 1],
                              attrib.vertices[3 * index.vertex_index + 2]},
                      .color = {1.0f, 1.0f, 1.0f},
                      .texCoord = {
                          attrib.texcoords[2 * index.texcoord_index + 0],
                          1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                      }};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<uint32_t>(model.vertices.size());
          model.vertices.push_back(vertex);
        }

        model.indices.push_back(uniqueVertices[vertex]);
      }
    }

    models.push_back(model);
  }

  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
                depthFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
                depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat,
                                     vk::ImageAspectFlagBits::eDepth, 1);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
      auto props = physicalDevice.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == vk::ImageTiling::eOptimal &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint ||
           format == vk::Format::eD24UnormS8Uint;
  }

  void createTextureSamplers() {
    for (Texture &texture : textures) {
      createTextureSampler(texture);
    }
  }

  void createTextureSampler(Texture &texture) {
    auto properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .sType = vk::StructureType::eSamplerCreateInfo,
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(texture.mipLevels),
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    VK_CHECK(device.createSampler(&samplerInfo, nullptr, &texture.sampler));
  }

  void createTextureImageViews() {
    for (Texture &texture : textures) {
      createTextureImageView(texture);
    }
  }

  void createTextureImageView(Texture &texture) {
    texture.imageView =
        createImageView(texture.image, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageAspectFlagBits::eColor, texture.mipLevels);
  }

  vk::ImageView createImageView(vk::Image image, vk::Format format,
                                vk::ImageAspectFlags aspectFlags,
                                uint32_t mipLevels) {
    vk::ImageViewCreateInfo viewInfo{
        .sType = vk::StructureType::eImageViewCreateInfo,
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {.aspectMask = aspectFlags,
                             .baseMipLevel = 0,
                             .levelCount = mipLevels,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};

    vk::ImageView imageView;
    VK_CHECK(device.createImageView(&viewInfo, nullptr, &imageView));

    return imageView;
  }

  void createTextureImages() {
    Texture tex1 = createTextureImage(TEXTURE_PATH);
    Texture tex2 = createTextureImage("textures/difuso_flip_oscuro.jpg");
    textures.push_back(tex1);
    textures.push_back(tex2);
  }

  Texture createTextureImage(std::string path) {
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(path.c_str(), &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;
    Texture texture;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    uint32_t mipLevels = static_cast<uint32_t>(std::floor(
                             std::log2(std::max(texWidth, texHeight)))) +
                         1;

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    VK_CHECK(device.mapMemory(stagingBufferMemory, 0, imageSize,
                              vk::MemoryMapFlags(0), &data));
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    device.unmapMemory(stagingBufferMemory);

    stbi_image_free(pixels);

    vk::Image image;
    vk::DeviceMemory imageMemory;
    createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
                vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst |
                    vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eDeviceLocal, image, imageMemory);

    transitionImageLayout(image, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);

    generateMipmap(image, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight,
                   mipLevels);

    texture.mipLevels = mipLevels;
    texture.image = image;
    texture.imageMemory = imageMemory;

    return texture;
  }

  void createColorResources() {
    vk::Format colorFormat = swapChainImageFormat;

    createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
                colorFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransientAttachment |
                    vk::ImageUsageFlagBits::eColorAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
                colorImageMemory);
    colorImageView = createImageView(colorImage, colorFormat,
                                     vk::ImageAspectFlagBits::eColor, 1);
  }

  void generateMipmap(vk::Image image, vk::Format imageFormat, int32_t texWidth,
                      int32_t texHeight, uint32_t mipLevels) {
    auto formatProperties = physicalDevice.getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
      throw std::runtime_error(
          "texture image format does not support linear blitting!");
    }

    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{
        .sType = vk::StructureType::eImageMemoryBarrier,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlags(0), 0, nullptr, 0,
                                    nullptr, 1, &barrier);

      vk::ImageBlit blit{
          .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = i - 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
          .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = i,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};
      blit.srcOffsets[0] = {0, 0, 0};
      blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
      blit.dstOffsets[0] = {0, 0, 0};
      blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1,
                            mipHeight > 1 ? mipHeight / 2 : 1, 1};

      commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                              image, vk::ImageLayout::eTransferDstOptimal, 1,
                              &blit, vk::Filter::eLinear);

      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eFragmentShader,
                                    vk::DependencyFlags(0), 0, nullptr, 0,
                                    nullptr, 1, &barrier);

      if (mipWidth > 1)
        mipWidth /= 2;
      if (mipHeight > 1)
        mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::DependencyFlags(0), 0, nullptr, 0,
                                  nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                   vk::SampleCountFlagBits numSamples, vk::Format format,
                   vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                   vk::MemoryPropertyFlags properties, vk::Image &image,
                   vk::DeviceMemory &imageMemory) {
    vk::ImageCreateInfo imageInfo{
        .sType = vk::StructureType::eImageCreateInfo,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {.width = static_cast<uint32_t>(width),
                   .height = static_cast<uint32_t>(height),
                   .depth = 1},
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = numSamples,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined};

    VK_CHECK(device.createImage(&imageInfo, nullptr, &image));

    vk::MemoryRequirements memRequirements;
    device.getImageMemoryRequirements(image, &memRequirements);

    vk::MemoryAllocateInfo allocInfo{
        .sType = vk::StructureType::eMemoryAllocateInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties)};

    VK_CHECK(device.allocateMemory(&allocInfo, nullptr, &imageMemory));

    device.bindImageMemory(image, imageMemory, 0);
  }

  void transitionImageLayout(vk::Image image, vk::Format format,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout, uint32_t mipLevels) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{
        .sType = vk::StructureType::eImageMemoryBarrier,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eNone,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = mipLevels,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eNone;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage,
                                  vk::DependencyFlags(0), 0, nullptr, 0,
                                  nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width,
                         uint32_t height) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = width, .height = height, .depth = 1}};

    commandBuffer.copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    endSingleTimeCommands(commandBuffer);
  }

  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .sType = vk::StructureType::eDescriptorSetAllocateInfo,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()};

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    VK_CHECK(device.allocateDescriptorSets(&allocInfo, descriptorSets.data()));

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i],
                                          .offset = 0,
                                          .range = sizeof(UniformBufferObject)};

      std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

      descriptorWrites[0].sType = vk::StructureType::eWriteDescriptorSet;
      descriptorWrites[0].dstSet = descriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      std::vector<vk::DescriptorImageInfo> imageInfos;

      for (Texture texture : textures) {
        vk::DescriptorImageInfo imageInfo{
            .sampler = texture.sampler,
            .imageView = texture.imageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        imageInfos.push_back(imageInfo);
      }

      descriptorWrites[1].sType = vk::StructureType::eWriteDescriptorSet;
      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType =
          vk::DescriptorType::eCombinedImageSampler;
      descriptorWrites[1].descriptorCount = (uint32_t)imageInfos.size();
      descriptorWrites[1].pImageInfo = imageInfos.data();

      device.updateDescriptorSets(
          static_cast<uint32_t>(descriptorWrites.size()),
          descriptorWrites.data(), 0, nullptr);
    }
  }

  void createImGuiDescriptorPool() {
    vk::DescriptorPoolSize poolSizes[] = {
        {vk::DescriptorType::eCombinedImageSampler,
         IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE},
    };

    vk::DescriptorPoolCreateInfo poolInfo = {};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 0;

    for (vk::DescriptorPoolSize &poolSize : poolSizes)
      poolInfo.maxSets += poolSize.descriptorCount;

    poolInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
    poolInfo.pPoolSizes = poolSizes;

    VK_CHECK(
        device.createDescriptorPool(&poolInfo, nullptr, &imGuiDescriptorPool));
  }

  void createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo{
        .sType = vk::StructureType::eDescriptorPoolCreateInfo,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};

    VK_CHECK(device.createDescriptorPool(&poolInfo, nullptr, &descriptorPool));
  }

  void createComputeDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2;

    vk::DescriptorPoolCreateInfo poolInfo{
        .sType = vk::StructureType::eDescriptorPoolCreateInfo,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};

    VK_CHECK(device.createDescriptorPool(&poolInfo, nullptr,
                                         &computeDescriptorPool));
  }

  void createUniformBuffers() {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible |
                       vk::MemoryPropertyFlagBits::eHostCoherent,
                   uniformBuffers[i], uniformBuffersMemory[i]);

      VK_CHECK(device.mapMemory(uniformBuffersMemory[i], 0, bufferSize,
                                vk::MemoryMapFlags(0),
                                &uniformBuffersMapped[i]));
    }
  }

  void createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .pImmutableSamplers = nullptr};

    vk::DescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 2, // TODO: CHANGE TO NUMBER OF TEXTURES
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr};

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};

    VK_CHECK(device.createDescriptorSetLayout(&layoutInfo, nullptr,
                                              &descriptorSetLayout));
  }

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                    vk::DeviceMemory &bufferMemory) {
    vk::BufferCreateInfo bufferInfo{.sType =
                                        vk::StructureType::eBufferCreateInfo,
                                    .size = size,
                                    .usage = usage,
                                    .sharingMode = vk::SharingMode::eExclusive};

    VK_CHECK(device.createBuffer(&bufferInfo, nullptr, &buffer));

    vk::MemoryRequirements memRequirements;
    device.getBufferMemoryRequirements(buffer, &memRequirements);

    vk::MemoryAllocateInfo allocInfo{
        .sType = vk::StructureType::eMemoryAllocateInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent)};

    VK_CHECK(device.allocateMemory(&allocInfo, nullptr, &bufferMemory));

    device.bindBufferMemory(buffer, bufferMemory, 0);
  }

  void createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    VK_CHECK(device.mapMemory(stagingBufferMemory, 0, bufferSize,
                              vk::MemoryMapFlags(0), &data));
    memcpy(data, indices.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
                 indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    VK_CHECK(device.mapMemory(stagingBufferMemory, 0, bufferSize,
                              vk::MemoryMapFlags(0), &data));
    memcpy(data, vertices.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
                 vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  vk::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1};

    vk::CommandBuffer commandBuffer;
    VK_CHECK(device.allocateCommandBuffers(&allocInfo, &commandBuffer));

    vk::CommandBufferBeginInfo beginInfo{
        .sType = vk::StructureType::eCommandBufferBeginInfo,
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    VK_CHECK(commandBuffer.begin(&beginInfo));

    return commandBuffer;
  }

  void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &commandBuffer};

    VK_CHECK(graphicsQueue.submit(1, &submitInfo, VK_NULL_HANDLE));
    graphicsQueue.waitIdle();

    device.freeCommandBuffers(commandPool, 1, &commandBuffer);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties;
    physicalDevice.getMemoryProperties(&memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo{
        .sType = vk::StructureType::eSemaphoreCreateInfo};

    vk::FenceCreateInfo fenceInfo{.sType = vk::StructureType::eFenceCreateInfo,
                                  .flags = vk::FenceCreateFlagBits::eSignaled};

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      VK_CHECK(device.createSemaphore(&semaphoreInfo, nullptr,
                                      &imageAvailableSemaphores[i]));
      VK_CHECK(device.createSemaphore(&semaphoreInfo, nullptr,
                                      &renderFinishedSemaphores[i]));
      VK_CHECK(device.createFence(&fenceInfo, nullptr, &inFlightFences[i]));

      VK_CHECK(device.createSemaphore(&semaphoreInfo, nullptr,
                                      &computeFinishedSemaphores[i]));
      VK_CHECK(
          device.createFence(&fenceInfo, nullptr, &computeInFlightFences[i]));
    }
  }

  void createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    vk::CommandBufferAllocateInfo allocInfo{
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = (uint32_t)commandBuffers.size()};

    VK_CHECK(device.allocateCommandBuffers(&allocInfo, commandBuffers.data()));
  }

  void recordCommandBuffer(vk::CommandBuffer commandBuffer,
                           uint32_t imageIndex) {
    vk::CommandBufferBeginInfo beginInfo{
        .sType = vk::StructureType::eCommandBufferBeginInfo};

    VK_CHECK(commandBuffer.begin(&beginInfo));

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color =
        vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    vk::RenderPassBeginInfo renderPassInfo{
        .sType = vk::StructureType::eRenderPassBeginInfo,
        .renderPass = renderPass,
        .framebuffer = swapChainFramebuffers[imageIndex],
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()};

    commandBuffer.beginRenderPass(&renderPassInfo,
                                  vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               graphicsPipeline);

    drawGeometry(commandBuffer);

    // commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
    // particlesPipeline);

    // drawParticles(commandBuffer);

    renderImGui(commandBuffer);

    commandBuffer.endRenderPass();

    commandBuffer.end();
  }

  void renderImGui(vk::CommandBuffer commandBuffer) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::DragFloat("FoV", &camera.fov);
    ImGui::DragFloat3("Eye", (float *)&camera.eye, 0.1f);
    ImGui::DragFloat3("Center", (float *)&camera.center, 0.1f);
    ImGui::DragFloat3("Up", (float *)&camera.up, 0.1f);

    uint32_t i = 0;
    for (Model &model : models) {
      ImGui::DragFloat3(std::format("Model {} pos", i).c_str(),
                        (float *)&model.position, 0.1f);
      ImGui::PushItemWidth(ImGui::CalcItemWidth() / 2);
      ImGui::DragFloat(std::format("##X{}", i).c_str(), &model.angleX, 1.0f,
                       0.0f, 0.0f, "x:%.1f");
      ImGui::SameLine();
      ImGui::DragFloat(std::format("##Y{}", i).c_str(), &model.angleY, 1.0f,
                       0.0f, 0.0f, "y:%.1f");
      ImGui::PopItemWidth();
      i++;
    }

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
  }

  void drawParticles(vk::CommandBuffer commandBuffer) {
    vk::Viewport viewport{.x = 0.0f,
                          .y = 0.0f,
                          .width = static_cast<float>(swapChainExtent.width),
                          .height = static_cast<float>(swapChainExtent.height),
                          .minDepth = 0.0f,
                          .maxDepth = 1.0f};
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};
    commandBuffer.setScissor(0, 1, &scissor);

    vk::DeviceSize particlesOffsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, &shaderStorageBuffers[currentFrame],
                                    particlesOffsets);

    commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
  }

  void drawGeometry(vk::CommandBuffer commandBuffer) {
    for (Model model : models) {
      vk::Viewport viewport{.x = 0.0f,
                            .y = 0.0f,
                            .width = static_cast<float>(swapChainExtent.width),
                            .height =
                                static_cast<float>(swapChainExtent.height),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f};
      commandBuffer.setViewport(0, 1, &viewport);

      vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};
      commandBuffer.setScissor(0, 1, &scissor);

      vk::DeviceSize vertexOffsets[] = {model.verticesOffset};
      vk::Buffer vertexBuffers[] = {vertexBuffer};
      commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, vertexOffsets);

      commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

      commandBuffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, graphicsPipelineLayout, 0, 1,
          &descriptorSets[currentFrame], 0, nullptr);

      ModelPushConstants constants;
      constants.model =
          glm::rotate(glm::mat4(1.0f), glm::radians(model.angleX),
                      glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)));
      constants.model =
          glm::rotate(constants.model, glm::radians(model.angleY),
                      glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)));
      constants.model = glm::translate(constants.model, model.position);

      commandBuffer.pushConstants(graphicsPipelineLayout,
                                  vk::ShaderStageFlagBits::eVertex, 0,
                                  sizeof(ModelPushConstants), &constants);

      commandBuffer.drawIndexed(static_cast<uint32_t>(model.indices.size()), 1,
                                model.indicesOffset, 0, 0);
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo poolInfo{
        .sType = vk::StructureType::eCommandPoolCreateInfo,
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex =
            queueFamilyIndices.graphicsAndComputeFamily.value()};

    VK_CHECK(device.createCommandPool(&poolInfo, nullptr, &commandPool));
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      std::array<vk::ImageView, 3> attachments = {
          colorImageView, depthImageView, swapChainImageViews[i]};

      vk::FramebufferCreateInfo framebufferInfo{
          .sType = vk::StructureType::eFramebufferCreateInfo,
          .renderPass = renderPass,
          .attachmentCount = static_cast<uint32_t>(attachments.size()),
          .pAttachments = attachments.data(),
          .width = swapChainExtent.width,
          .height = swapChainExtent.height,
          .layers = 1};

      VK_CHECK(device.createFramebuffer(&framebufferInfo, nullptr,
                                        &swapChainFramebuffers[i]));
    }
  }

  void createRenderPass() {
    vk::AttachmentDescription colorAttachment{
        .format = swapChainImageFormat,
        .samples = msaaSamples,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal};

    vk::AttachmentReference colorAttachmentRef{
        .attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

    vk::AttachmentDescription colorAttachmentResolve{
        .format = swapChainImageFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR};

    vk::AttachmentReference colorAttachmentResolveRef{
        .attachment = 2, .layout = vk::ImageLayout::eColorAttachmentOptimal};

    vk::AttachmentDescription depthAttachment{
        .format = findDepthFormat(),
        .samples = msaaSamples,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    vk::AttachmentReference depthAttachmentRef{
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = &colorAttachmentResolveRef,
        .pDepthStencilAttachment = &depthAttachmentRef,
    };

    vk::SubpassDependency dependency{
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite};

    std::array<vk::AttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, colorAttachmentResolve};
    vk::RenderPassCreateInfo renderPassInfo{
        .sType = vk::StructureType::eRenderPassCreateInfo,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency};

    VK_CHECK(device.createRenderPass(&renderPassInfo, nullptr, &renderPass));
  }

  void createParticlePipeline() {
    auto vertShaderCode = readFile("shaders/bin/particle.vert.spv");
    auto fragShaderCode = readFile("shaders/bin/particle.frag.spv");

    vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertShaderModule,
        .pName = "main"};

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragShaderModule,
        .pName = "main"};

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                        fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
        .topology = vk::PrimitiveTopology::ePointList,
        .primitiveRestartEnable = vk::False};

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)swapChainExtent.width,
        .height = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};

    vk::PipelineViewportStateCreateInfo viewportState{
        .sType = vk::StructureType::ePipelineViewportStateCreateInfo,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::True, // enable sample shading
        .minSampleShading = 0.2f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = vk::False,
        .alphaToOneEnable = vk::False,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .sType = vk::StructureType::ePipelineDepthStencilStateCreateInfo,
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr};

    VK_CHECK(device.createPipelineLayout(&pipelineLayoutInfo, nullptr,
                                         &particlesPipelineLayout));

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = particlesPipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE};

    VK_CHECK(device.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipelineInfo,
                                            nullptr, &particlesPipeline));

    device.destroyShaderModule(fragShaderModule);
    device.destroyShaderModule(vertShaderModule);
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/bin/model.vert.spv");
    auto fragShaderCode = readFile("shaders/bin/model.frag.spv");

    vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertShaderModule,
        .pName = "main"};

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragShaderModule,
        .pName = "main"};

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                        fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False};

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)swapChainExtent.width,
        .height = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};

    vk::PipelineViewportStateCreateInfo viewportState{
        .sType = vk::StructureType::ePipelineViewportStateCreateInfo,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::True, // enable sample shading
        .minSampleShading = 0.2f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = vk::False,
        .alphaToOneEnable = vk::False,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .sType = vk::StructureType::ePipelineDepthStencilStateCreateInfo,
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment};

    vk::PushConstantRange pushConstant{.stageFlags =
                                           vk::ShaderStageFlagBits::eVertex,
                                       .offset = 0,
                                       .size = sizeof(ModelPushConstants)};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant};

    VK_CHECK(device.createPipelineLayout(&pipelineLayoutInfo, nullptr,
                                         &graphicsPipelineLayout));

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = graphicsPipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE};

    VK_CHECK(device.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipelineInfo,
                                            nullptr, &graphicsPipeline));

    device.destroyShaderModule(fragShaderModule);
    device.destroyShaderModule(vertShaderModule);
  }

  vk::ShaderModule createShaderModule(const std::vector<char> &code) {
    vk::ShaderModuleCreateInfo createInfo{
        .sType = vk::StructureType::eShaderModuleCreateInfo,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data())};

    vk::ShaderModule shaderModule;
    VK_CHECK(device.createShaderModule(&createInfo, nullptr, &shaderModule));

    return shaderModule;
  }

  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
      swapChainImageViews[i] =
          createImageView(swapChainImages[i], swapChainImageFormat,
                          vk::ImageAspectFlagBits::eColor, 1);
    }
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{
        .sType = vk::StructureType::eSwapchainCreateInfoKHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment};

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsAndComputeFamily.value(),
                                     indices.presentFamily.value()};

    if (indices.graphicsAndComputeFamily != indices.presentFamily) {
      createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = vk::True;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(device.createSwapchainKHR(&createInfo, nullptr, &swapChain));

    auto swapChainImagesKhr = device.getSwapchainImagesKHR(swapChain);
    swapChainImages.resize(swapChainImagesKhr.size());
    swapChainImages = device.getSwapchainImagesKHR(swapChain);

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void createSurface() {
    VK_CHECK(glfwCreateWindowSurface((VkInstance)instance, window, nullptr,
                                     (VkSurfaceKHR *)&surface));
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsAndComputeFamily.value(),
        indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo{
          .sType = vk::StructureType::eDeviceQueueCreateInfo,
          .queueFamilyIndex = indices.graphicsAndComputeFamily.value(),
          .queueCount = 1,
          .pQueuePriorities = &queuePriority};
      queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{
        .sampleRateShading = vk::True, // enable sample shading
        .samplerAnisotropy = vk::True};

    vk::PhysicalDeviceSynchronization2Features syncFeature{.synchronization2 =
                                                               true};

    vk::DeviceCreateInfo createInfo{
        .sType = vk::StructureType::eDeviceCreateInfo,
        .pNext = &syncFeature,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .ppEnabledLayerNames = deviceExtensions.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures,
    };

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    VK_CHECK(physicalDevice.createDevice(&createInfo, nullptr, &device));

    graphicsQueue =
        device.getQueue(indices.graphicsAndComputeFamily.value(), 0);
    computeQueue = device.getQueue(indices.graphicsAndComputeFamily.value(), 0);
    presentQueue = device.getQueue(indices.presentFamily.value(), 0);
  }

  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };

  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) {
    SwapChainSupportDetails details;

    VK_CHECK(device.getSurfaceCapabilitiesKHR(surface, &details.capabilities));

    auto surfaceFormats = device.getSurfaceFormatsKHR(surface);
    if (surfaceFormats.size() != 0) {
      details.formats.resize(surfaceFormats.size());
      details.formats = device.getSurfaceFormatsKHR(surface);
    }

    auto surfacePresent = device.getSurfacePresentModesKHR(surface);
    if (surfacePresent.size() != 0) {
      details.presentModes.resize(surfacePresent.size());
      details.presentModes = device.getSurfacePresentModesKHR(surface);
    }

    return details;
  }

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsAndComputeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
      return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
  };

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;

    auto queueFamilies = device.getQueueFamilyProperties();

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) &&
          (queueFamily.queueFlags & vk::QueueFlagBits::eCompute)) {
        indices.graphicsAndComputeFamily = i;
      }

      vk::Bool32 presentSupport = false;
      VK_CHECK(device.getSurfaceSupportKHR(i, surface, &presentSupport));

      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  void pickPhysicalDevice() {
    auto devices = instance.enumeratePhysicalDevices();

    if (devices.size() == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        msaaSamples = getMaxUsableSampleCount();
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  bool isDeviceSuitable(vk::PhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    auto supportedFeatures = device.getFeatures();

    return indices.isComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
  }

  bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    auto availableExtensions = device.enumerateDeviceExtensionProperties();
    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
          availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes) {
    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
        return availablePresentMode;
      }
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D
  chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetFramebufferSize(window, &width, &height);

      vk::Extent2D actualExtent = {static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height)};

      actualExtent.width =
          std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                     capabilities.maxImageExtent.width);
      actualExtent.height =
          std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                     capabilities.maxImageExtent.height);

      return actualExtent;
    }
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                     &debugMessenger) != vk::Result::eSuccess) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    vk::ApplicationInfo applicationInfo{
        .sType = vk::StructureType::eApplicationInfo,
        .pApplicationName = "Tengine",
        .applicationVersion = vk::makeApiVersion(0, 1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = vk::makeApiVersion(0, 1, 0, 0),
        .apiVersion = vk::ApiVersion10};

    vk::InstanceCreateInfo instanceCreateInfo{
        .sType = vk::StructureType::eInstanceCreateInfo,
        .pApplicationInfo = &applicationInfo};

    auto extensions = getRequiredExtensions();
    instanceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
      instanceCreateInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();

      populateDebugMessengerCreateInfo(debugCreateInfo);
      instanceCreateInfo.pNext =
          (vk::DebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    } else {
      instanceCreateInfo.enabledLayerCount = 0;

      instanceCreateInfo.pNext = nullptr;
    }

    VK_CHECK(vk::createInstance(&instanceCreateInfo, nullptr, &instance));
  }

  void cleanupSwapChain() {
    device.destroyImageView(colorImageView);
    device.destroyImage(colorImage);
    device.freeMemory(colorImageMemory);

    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);

    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
      device.destroyFramebuffer(swapChainFramebuffers[i]);
    }

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      device.destroyImageView(swapChainImageViews[i]);
    }

    device.destroySwapchainKHR(swapChain);
  }

  void recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    device.waitIdle();

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createColorResources();
    createDepthResources();
    createFramebuffers();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }

    device.waitIdle();
  }

  void drawFrame() {
    vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo};

    // Compute submissions
    VK_CHECK(device.waitForFences(1, &computeInFlightFences[currentFrame],
                                  vk::True, UINT64_MAX));

    updateUniformBuffer(currentFrame);

    VK_CHECK(device.resetFences(1, &computeInFlightFences[currentFrame]));

    computeCommandBuffers[currentFrame].reset();
    recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

    VK_CHECK(computeQueue.submit(1, &submitInfo,
                                 computeInFlightFences[currentFrame]));

    // Graphics submissions
    VK_CHECK(device.waitForFences(1, &inFlightFences[currentFrame], vk::True,
                                  UINT64_MAX));

    uint32_t imageIndex;
    vk::Result result = device.acquireNextImageKHR(
        swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE, &imageIndex);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    } else if (result != vk::Result::eSuccess &&
               result != vk::Result::eSuboptimalKHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    // updateUniformBuffer(currentFrame);

    VK_CHECK(device.resetFences(1, &inFlightFences[currentFrame]));

    commandBuffers[currentFrame].reset();
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    vk::Semaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame],
                                      imageAvailableSemaphores[currentFrame]};
    vk::Semaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eVertexInput};

    submitInfo = {};
    submitInfo.sType = vk::StructureType::eSubmitInfo;
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VK_CHECK(
        graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]));

    vk::SwapchainKHR swapChains[] = {swapChain};

    vk::PresentInfoKHR presentInfo{.sType = vk::StructureType::ePresentInfoKHR,
                                   .waitSemaphoreCount = 1,
                                   .pWaitSemaphores =
                                       &renderFinishedSemaphores[currentFrame],
                                   .swapchainCount = 1,
                                   .pSwapchains = swapChains,
                                   .pImageIndices = &imageIndex,
                                   .pResults = nullptr};

    result = presentQueue.presentKHR(&presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void updateUniformBuffer(uint32_t currentImage) {
    UniformBufferObject ubo{
        .view = glm::lookAt(camera.eye, camera.center, camera.up),
        .proj = glm::perspective(
            glm::radians(camera.fov),
            swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f),
    };

    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }

  void cleanup() {
    cleanupSwapChain();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    for (Texture texture : textures) {
      device.destroySampler(texture.sampler);
      device.destroyImageView(texture.imageView);
      device.destroyImage(texture.image);
      device.freeMemory(texture.imageMemory);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      device.destroyBuffer(uniformBuffers[i]);
      device.freeMemory(uniformBuffersMemory[i]);

      device.destroyBuffer(shaderStorageBuffers[i]);
      device.freeMemory(shaderStorageBuffersMemory[i]);
    }

    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorPool(imGuiDescriptorPool);
    device.destroyDescriptorPool(computeDescriptorPool);

    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyDescriptorSetLayout(computeDescriptorSetLayout);

    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);

    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);

    device.destroyPipeline(particlesPipeline);
    device.destroyPipelineLayout(particlesPipelineLayout);

    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(graphicsPipelineLayout);

    device.destroyPipeline(computePipeline);
    device.destroyPipelineLayout(computePipelineLayout);

    device.destroyRenderPass(renderPass);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      device.destroySemaphore(imageAvailableSemaphores[i]);
      device.destroySemaphore(renderFinishedSemaphores[i]);
      device.destroyFence(inFlightFences[i]);

      device.destroySemaphore(computeFinishedSemaphores[i]);
      device.destroyFence(computeInFlightFences[i]);
    }

    device.destroyCommandPool(commandPool);

    device.destroy();

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    instance.destroySurfaceKHR(surface);
    instance.destroy();

    glfwDestroyWindow(window);

    glfwTerminate();
  }

  static std::vector<char> readFile(const std::string &fileName) {
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL
  debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                vk::DebugUtilsMessageTypeFlagsEXT messageType,
                const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return vk::False;
  }
};

int main() {
  TengineApp app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
