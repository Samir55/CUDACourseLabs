cmake_minimum_required(VERSION 3.10)
project(ModuleThreeLabs)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_61,code=sm_61
)
# Specify include directories
include_directories(
        src/ImageBlurKernel
        src/ColorToGreyscaleKernel
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        moduleThreeLabs
        src/main.cpp
        src/ImageBlurKernel/image_blur_kernel.h
        src/ImageBlurKernel/image_blur_kernel.cu
        src/ColorToGreyscaleKernel/color_to_greyscale_kernel.h
        src/ColorToGreyscaleKernel/color_to_greyscale_kernel.cu
        src/Utils/utils.h
)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
        moduleThreeLabs
        ${OpenCV_LIBS}
)

