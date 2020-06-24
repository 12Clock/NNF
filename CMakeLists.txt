cmake_minimum_required(VERSION 2.8)

project(nnf)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

# check results and add flag
if(COMPILER_SUPPORTS_CXX11)#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)#
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# set(NNF_INSTUDE_DIR ${PROJECT_SOURCE_DIR}/NNFT)

# ADD_SUBDIRECTORY(src/FTen)

set(SOURCES 
# src/NNFBase/NNFBaseMacros.h
src/NNFBase/utils/NNFException.cpp
# src/NNFBase/utils/NNFException.h
# src/NNFBase/utils/NNFStringUtils.h
# src/NNFBase/utils/NNFExceptionMacros.h
)

add_library(hello_shared_library
    SHARED ${SOURCES}
)

add_library(hello::library ALIAS hello_shared_library)

target_include_directories(hello_shared_library
    PUBLIC
        ${PROJECT_SOURCE_DIR}
)

add_executable(hello_shared_binary
    main.cpp
)

target_link_libraries(hello_shared_binary
    PRIVATE
        hello::library
)
