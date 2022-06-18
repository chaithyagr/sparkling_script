# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(bbfmm)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

# Gather all source files
set(sources  BBFMM2D/src/H2_2D_Tree.cpp
             BBFMM2D/src/H2_2D_Node.cpp
             BBFMM2D/src/kernel_Base.cpp
             BBFMM2D/src/kernel_Types.cpp
             BBFMM_Repulsion.cpp)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3")

#This is for Eigen
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(bbfmm SHARED ${sources})
set_target_properties(bbfmm PROPERTIES PREFIX "")

include_directories(
    BBFMM2D/header/
    ${PYTHON_INCLUDE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    eigen
)
