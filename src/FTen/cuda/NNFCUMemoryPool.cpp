#ifndef NNFCU_MEMORY_POOL_CPP
#define NNFCU_MEMORY_POOL_CPP

#include "NNFCUMemoryPool.h"

namespace nnf {

namespace cuda {

namespace cache {

static bool CUDABlockComparator(const CUDABlock* a, const CUDABlock* b)
{
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static size_t round_size(const size_t size)
{
    if(size < kMinBlockSize){
        return kMinBlockSize;
    }else{
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}

} // namespace cache

} // namespace cuda

} // namespace nnf

#endif