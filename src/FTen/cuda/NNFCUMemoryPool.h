#ifndef NNFCU_MEMORY_POOL_H
#define NNFCU_MEMORY_POOL_H

#include <cuda_runtime_api.h>
#include <unordered_set>

namespace cuda{

/*
*/

class CUDAStream final
{
    private:
        int16_t DeviceIndex;
        int32_t StreamId;
    public:
        enum Unsafe { UNSAFE };
        enum Default { DEFAULT };

        explicit CUDAStream(Unsafe, int16_t index, int32_t id): DeviceIndex(index), StreamId(id) {}
        explicit CUDAStream(Default, int16_t index): DeviceIndex(index), StreamId(0) {}

        int16_t device_index() const noexcept;
        int32_t stream_id() const noexcept;

        bool operator==(const CUDAStream &other) const noexcept;
        bool operator!=(const CUDAStream &other) const noexcept;
};

using stream_set = std::unordered_set<CUDAStream>;

namespace cache{

struct CUDABlock
{
    int16_t       device;      // gpu
    cudaStream_t  stream;      // allocation stream
    stream_set    stream_uses; // streams on which the block was used
    size_t        size;        // block size in bytes
    void*         ptr;         // memory address
    bool          allocated;   // in-use flag      
    int32_t       event_count; // number of outstanding CUDA events
    
    CUDABlock*        prev;        // prev block if split from a larger allocation
    CUDABlock*        next;        // next block if split from a larger allocation

    CUDABlock(int16_t device, cudaStream_t stream, size_t size, void *ptr):
    device(device), stream(stream), stream_uses(), size(size), ptr(ptr), 
    allocated(false), event_count(0), prev(nullptr), next(nullptr) {}

    CUDABlock(int16_t device, cudaStream_t stream, size_t size):
    device(device), stream(stream), stream_uses(), size(size), ptr(nullptr), 
    allocated(false), event_count(0), prev(nullptr), next(nullptr) {}
};

} // namespace cache

} // namespace cuda

#endif