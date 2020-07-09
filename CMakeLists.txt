cmake_minimum_required(VERSION 2.8)

project(nnf)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

# check results and add flag
if(COMPILER_SUPPORTS_CXX14)#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# elseif(COMPILER_SUPPORTS_CXX0X)#
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

find_package(CUDA)
include_directories(
    "${CUDA_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}"
)

add_subdirectory(src/FTen)

set(EXTRA_LIBS 
    ${EXTRA_LIBS} 
    ${FTEN_LIBS}
)

add_executable(test_nnf
    main.cpp
)

target_link_libraries(test_nnf
    ${EXTRA_LIBS}
    ${CUDA_LIBRARIES}
)
