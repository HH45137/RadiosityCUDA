cmake_minimum_required(VERSION 4.2)
project(RadiosityCUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Debug device and global function is must have
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G --generate-line-info")

find_package(glm REQUIRED)

add_executable(${PROJECT_NAME}
        main.cu
        common.cuh
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)
