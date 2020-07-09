#ifndef NNFC_ALLOCATOR_H
#define NNFC_ALLOCATOR_H

#include <src/FTen/cpu/NNFCMemoryPool.h>
#include <src/FTen/cpu/NNFCMemoryPool.cpp>
#include <mutex>
#include <bitset>

namespace nnf {

namespace utils {

class OutOfMemoryError: nnf::utils::NNF_Error{
    using NNF_Error::NNF_Error;
};

} // namespace utils

namespace cpu {

namespace cache {

enum struct StatType : uint64_t {
    AGGREGATE = 0,
    FIRST_POOL = 1,
    SECOND_POOL = 2,
    THIRD_POOL = 3,
    NUM_TYPES = 4  // remember to update this whenever a new stat type is added
};

typedef std::bitset<static_cast<size_t>(StatType::NUM_TYPES)> StatTypes;
typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
    StatArray active;                // COUNT: number of active memory blocks (allocated)
    StatArray inactive;              // COUNT: number of inactive, split memory blocks (unallocated but can't be released via cudaFree)

    StatArray active_bytes;          // SUM: bytes within active memory blocks
    StatArray inactive_bytes;        // SUM: bytes within inactive, split memory blocks
};

class HostAllocator
{
private:
    mutable std::recursive_mutex mutex[3];     // Each Pool has a mutex
    DeviceStats stats;

    MapBlockPool first_pool = MapBlockPool(PoolType::FIRST);                    // First Pool, unallocated cached blocks smaller than 4 KiB
    SetBlockPool second_pool = SetBlockPool(PoolType::SECOND);                  // Second Pool, unallocated cached blocks smaller than 2 MiB
    SetBlockPool third_pool = SetBlockPool(PoolType::THIRD);                    // Third Pool, unallocated cached blocks larger than 2 MiB

    template<class T>
    Block* malloc_with_pool(size_t size, T& pool);
    template<class T>
    StatType get_stat_type_for_pool(T& pool);
    template<class T>
    bool free_blocks_with_pool(T& pool, int idx, PoolStats& pool_stats);
    bool free_cache_blocks();
    template<class T>
    void free_with_pool(Block* block, T& pool);
public:
    Block* malloc(size_t size);
    void free(Block* block);
    void* getBaseAllocation(Block *block, size_t *outSize)=delete;

    // Get the current device status
    DeviceStats getStats() {
        std::lock_guard<std::recursive_mutex> lock1(mutex[0]);
        std::lock_guard<std::recursive_mutex> lock2(mutex[1]);
        std::lock_guard<std::recursive_mutex> lock3(mutex[2]);
        return stats;
    }
    // Empty all chahe
    void emptyCache() {
        std::lock_guard<std::recursive_mutex> lock1(mutex[0]);
        std::lock_guard<std::recursive_mutex> lock2(mutex[1]);
        std::lock_guard<std::recursive_mutex> lock3(mutex[2]);
        free_cache_blocks();
    }

    // Try empty all chahe
    void tryEmptyCache() {
        free_cache_blocks();
    }
    
    // Resets the historical accumulated stats for the device
    void resetAccumulatedStats() {
        std::lock_guard<std::recursive_mutex> lock1(mutex[0]);
        std::lock_guard<std::recursive_mutex> lock2(mutex[1]);
        std::lock_guard<std::recursive_mutex> lock3(mutex[2]);

        for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
            reset_accumulated_stat(stats.active[statType]);
            reset_accumulated_stat(stats.inactive[statType]);
            reset_accumulated_stat(stats.active_bytes[statType]);
            reset_accumulated_stat(stats.inactive_bytes[statType]);
        }
    }

    // Resets the historical peak stats for the device
    void resetPeakStats() {
        std::lock_guard<std::recursive_mutex> lock1(mutex[0]);
        std::lock_guard<std::recursive_mutex> lock2(mutex[1]);
        std::lock_guard<std::recursive_mutex> lock3(mutex[2]);

        for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
            reset_peak_stat(stats.active[statType]);
            reset_peak_stat(stats.inactive[statType]);
            reset_peak_stat(stats.active_bytes[statType]);
            reset_peak_stat(stats.inactive_bytes[statType]);
        }
    }
};

} // namespace cache

} // namespace cpu

} // namespace nnf

#endif