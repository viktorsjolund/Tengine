# Set vulkan layer path
export VK_ADD_LAYER_PATH="$PWD/build/vcpkg_installed/x64-windows/bin"

# Compile shaders
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/model.vert -o shaders/bin/model.vert.spv
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/model.frag -o shaders/bin/model.frag.spv
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/particle.vert -o shaders/bin/particle.vert.spv
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/particle.frag -o shaders/bin/particle.frag.spv
./build/vcpkg_installed/x64-windows/tools/shaderc/glslc.exe shaders/compute.comp -o shaders/bin/compute.comp.spv

# Compile application
cmake -DCMAKE_BUILD_TYPE=Debug --preset=vcpkg
cmake --build build
