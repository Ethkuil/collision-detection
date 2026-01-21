cmake_minimum_required(VERSION 3.18)
project(collision-detection CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(collision_detection
    main.cc
    collision_detection.cu
    collision_detection.cuh
    renderer.cc
    renderer.h
)

target_link_libraries(collision_detection PRIVATE 
    CUDA::cudart
    OpenGL::OpenGL
    GLUT::GLUT
)
