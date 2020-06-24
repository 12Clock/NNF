#ifndef NNFCU_MACROS_H
#define NNFCU_MACROS_H

#include <src/NNFBase/NNFBaseMacros.h>
#include <cuda.h>

#define NNFCU_CHECK(EXPR)                                                     \
    do {                                                                      \
        cudaError_t _err = EXPR;                                              \
        if(_err != cudaSuccess){                                              \
            auto unused_err NNF_UNUSED = cudaGetLastError();                  \
            NNF_CHECK(false, "CUDA error : ", cudaGetErrorString(_err));      \
        }                                                                     \
    }while(0)

#endif