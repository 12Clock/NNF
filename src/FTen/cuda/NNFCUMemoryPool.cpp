#ifndef NNFCU_MEMORY_POOL_CPP
#define NNFCU_MEMORY_POOL_CPP

#include "NNFCUMemoryPool.h"

namespace cuda
{

int16_t CUDAStream::device_index() const noexcept
{
    return DeviceIndex;
}

int32_t CUDAStream::stream_id() const noexcept
{
    return StreamId;
}

bool CUDAStream::operator==(const CUDAStream &other) const noexcept
{
    return this->DeviceIndex == other.device_index() && this->StreamId == other.stream_id();
}

bool CUDAStream::operator!=(const CUDAStream &other) const noexcept
{
    return !(*this == other);
}

namespace cache
{

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

} // namespace cache

} // namespace cuda

#endif