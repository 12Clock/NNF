#ifndef NNFCU_ALLOCATOR_H
#define NNFCU_ALLOCATOR_H

#include "NNFCUMemoryPool.h"
#include "NNFCUMemoryPool.cpp"
#include <set>
#include <bitset>
#include <mutex>
#include <deque>
#include <utility>

/*
Code reference pytorch, github: https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp
*/
namespace nnf{

namespace utils{

class CUDAOutOfMemoryError: nnf::utils::NNF_Error{
  using NNF_Error::NNF_Error;

  
};

} // namespace utils

} // namespace nnf

namespace cuda
{

namespace cache
{

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3  // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  StatArray allocation;            // COUNT: allocations requested by client code
  StatArray segment;               // COUNT: number of allocated segments from cudaMalloc().
  StatArray active;                // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray inactive_split;        // COUNT: number of inactive, split memory blocks (unallocated but can't be released via cudaFree)

  StatArray allocated_bytes;       // SUM: bytes requested by client code
  StatArray reserved_bytes;        // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray active_bytes;          // SUM: bytes within active memory blocks
  StatArray inactive_split_bytes;  // SUM: bytes within inactive, split memory blocks

  int64_t num_alloc_retries = 0;   // COUNT: total number of failed calls to CUDA malloc necessitating cache flushes.

  int64_t num_ooms = 0;            // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
};

typedef bool (*Comparator)(const CUDABlock *, const CUDABlock *);
typedef std::set<CUDABlock *, Comparator> CUDABlockPool;

typedef std::bitset<static_cast<size_t>(StatType::NUM_TYPES)> StatTypes;

struct AllocParams {

  CUDABlock search_key; //Search keywords, instantiated as block
  CUDABlockPool *pool;  //large_blocks or small_blocks
  size_t alloc_size;    //Allocated size
  CUDABlock *block;     //Allocated block
  StatTypes stat_types;
  cudaError_t err;

  AllocParams(int device, size_t size, cudaStream_t stream, CUDABlockPool *pool, size_t alloc_size,
              DeviceStats &stats) :
    search_key(device, stream, size, pool),
    pool(pool),
    alloc_size(alloc_size),
    block(nullptr),
    err(cudaSuccess) {}

  int device() { return search_key.device; }
  cudaStream_t stream() { return search_key.stream; }
  size_t size() { return search_key.size; }
};

class DeviceAllocator 
{
  private:
    mutable std::recursive_mutex mutex;                          // Lock around all operations
    DeviceStats stats;                                           // Device statistics

    CUDABlockPool large_blocks;                                  // Large memory pool, unallocated cached blocks larger than 1 MB
    CUDABlockPool small_blocks;                                  // Small memory pool, unallocated cached blocks 1 MB or smaller

    std::unordered_set<CUDABlock*> active_blocks;                // Allocated or in use by a stream
    std::deque<std::pair<cudaEvent_t, CUDABlock*> > cuda_events; // Outstanding cuda events

    //----------------------------------function-----------------------------------------

    size_t try_merge_blocks(CUDABlock* dst, 
                             CUDABlock* src, 
                             CUDABlockPool& pool); // 
    StatType get_stat_type_for_pool(const CUDABlockPool& pool) const;
    void free_block(CUDABlock* block);      //
    void process_events();                  // Process outstanding cudaEvents
    CUDABlockPool& get_pool(size_t size);
    static size_t get_allocation_size(const size_t size);
    bool get_free_block(AllocParams& params);
    bool trigger_free_memory_callbacks(AllocParams& params); // recycle free memory by Gabage Collector(GC)
    bool alloc_block(AllocParams& params, bool isRetry);
    void synchronize_and_free_events();
    void free_blocks(CUDABlockPool& pool);
    bool free_cached_blocks();
    static std::string format_size(size_t size);
    bool should_split(CUDABlock* block, size_t size);
    void insert_events(CUDABlock* block);
    cudaEvent_t create_event_internal();

  public:
    DeviceAllocator(): large_blocks(CUDABlockComparator), small_blocks(CUDABlockComparator) {} //function CUDABlockComparator is from NNFCUMemoryPool.cpp

    CUDABlock* malloc(int device, size_t size, cudaStream_t stream);
    void free(CUDABlock *block);
    void* getBaseAllocation(CUDABlock *block, size_t *outSize);
    void recordStream(CUDABlock *block, cuda::CUDAStream stream);
};



} // namespace cache

} // namespace cuda

#endif