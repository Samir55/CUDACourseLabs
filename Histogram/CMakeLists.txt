cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

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
        BasicMulitplicationKernel
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        MatrixMultiplicationExec
        src/main.cpp
        src/Histogram/Histogram.h
        src/Histogram/HistogramKernel.cu
        src/Histogram/HistogramKernel.h
        src/Utils/utils.h
)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
        MatrixMultiplicationExec
        ${OpenCV_LIBS}
)

