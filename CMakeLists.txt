cmake_minimum_required(VERSION 3.28.1)

project(Tengine)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

set(CMAKE_GENERATOR "MinGW Makefiles")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/textures DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

add_executable(Tengine src/main.cpp)

set_property(TARGET Tengine PROPERTY CXX_STANDARD 23)

target_include_directories(Tengine PRIVATE ${Stb_INCLUDE_DIR})

target_link_libraries(Tengine PRIVATE 
  Vulkan::Vulkan
  glfw
  Vulkan::Headers
  glm::glm
  tinyobjloader::tinyobjloader
)
