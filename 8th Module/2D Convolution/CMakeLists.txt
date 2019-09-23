cmake_minimum_required(VERSION 3.14)
project(templateProject)

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
        src/StencilKernel
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        templateProjectExec
        src/main.cpp
        src/StencilKernel/stencilKernel.cu
        src/StencilKernel/stencilKernel.h
        src/Utils/utils.h
        src/ConvolutionKernel/
)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
        templateProjectExec
        ${OpenCV_LIBS}
)

