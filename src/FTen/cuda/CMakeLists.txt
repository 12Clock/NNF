CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(gpu)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(gpu STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})