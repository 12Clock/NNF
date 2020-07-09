CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(FTen)

set(FTEN_LIBS ${FTEN_LIBS}
${CMAKE_CURRENT_SOURCE_DIR}/cpu/NNFCAllocator.cpp
${CMAKE_CURRENT_SOURCE_DIR}/cpu/NNFMemoryPool.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cuda")

add_subdirectory(cuda)
set(FTEN_LIBS ${FTEN_LIBS} gpu)
