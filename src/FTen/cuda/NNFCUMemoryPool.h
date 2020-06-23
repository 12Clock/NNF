#ifndef NNFCU_MEMORY_POOL_H
#define NNFCU_MEMORY_POOL_H

#include <cuda_runtime_api.h>
#include <unordered_set>

namespace cuda{

/*
Code reference pytorch, github: https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp
*/

class CUDAStream final
{
    private:
        int16_t DeviceIndex; //GPU device index, like "cuda:0"
        int32_t StreamId; //Cuda stream id
    public:
        enum Unsafe { UNSAFE };
        enum Default { DEFAULT };

        explicit CUDAStream(Unsafe, int16_t index, int32_t id): DeviceIndex(index), StreamId(id) {}
        explicit CUDAStream(Default, int16_t index): DeviceIndex(index), StreamId(0) {}

        int16_t device_index() const noexcept;
        int32_t stream_id() const noexcept;
        cudaStream_t stream() const;

        bool operator==(const CUDAStream &other) const noexcept;
        bool operator!=(const CUDAStream &other) const noexcept;
};

using stream_set = std::unordered_set<CUDAStream>;

namespace cache{

constexpr size_t kMinBlockSize = 512;       // All sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // Largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // Allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // Round up large allocs to 2 MiB

struct CUDABlock
{
    int16_t        device;      // Gpu
    cudaStream_t   stream;      // Allocation stream
    stream_set     stream_uses; // Streams on which the block was used
    size_t         size;        // Block size in bytes
    CUDABlockPool* pool;
    void*          ptr;         // Memory address
    bool           allocated;   // In-use flag      
    int32_t        event_count; // Number of outstanding CUDA events
    
    CUDABlock*        prev;        // Prev block if split from a larger allocation
    CUDABlock*        next;        // Next block if split from a larger allocation

    CUDABlock(int16_t device, cudaStream_t stream, size_t size, CUDABlockPool* pool, void *ptr):
    device(device), stream(stream), stream_uses(), size(size), pool(pool), ptr(ptr), 
    allocated(false), event_count(0), prev(nullptr), next(nullptr) {}

    CUDABlock(int16_t device, cudaStream_t stream, size_t size, CUDABlockPool* pool):
    device(device), stream(stream), stream_uses(), size(size), pool(pool), ptr(nullptr), 
    allocated(false), event_count(0), prev(nullptr), next(nullptr) {}

    bool is_split() const {
        return (prev != nullptr) || (next != nullptr);
    }
};

} // namespace cache

} // namespace cuda

#endif