# Set vulkan layer path
export VK_ADD_LAYER_PATH="$PWD/build/vcpkg_installed/x64-windows/bin"

# Compile shaders
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/shader.vert -o shaders/vert.spv
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/shader.frag -o shaders/frag.spv

# Compile application
cmake -DCMAKE_BUILD_TYPE=Debug --preset=vcpkg
cmake --build build
