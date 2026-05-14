cmake_minimum_required(VERSION 4.2)
project(RadiosityCUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(RadiosityCUDA main.cu
        OBJ_Loader.h
        math.cu
        obj_io.h
        common.h)

set_target_properties(RadiosityCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CUDA_FLAGS "--generate-line-info")

