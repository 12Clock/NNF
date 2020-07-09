#ifndef NNFC_ALLOCATOR_CPP
#define NNFC_ALLOCATOR_CPP

#include <src/FTen/cpu/NNFCAllocator.h>
#include <thread>

namespace nnf {

namespace cpu {

namespace cache {

static std::string format_size(uint64_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KiB";
    } else if (size <= 1073741824ULL) {
        os << size / 1048576.0;
        os << " MiB";
    } else {
        os << size / 1073741824.0;
        os << " GiB";
    }
    return os.str();
}

//-----------------------------------[struct Stat]----------------------------------------

void update_stat_by_stat(Stat& stat1, Stat& stat2) {
    stat1.allocated += stat2.allocated;
    stat1.current += stat2.current;
    stat1.freed += stat2.freed;
    stat1.peak += stat2.peak;
}

void update_stat_array(DeviceStats& device_stats, PoolStats& pool_stats, const StatTypes& stat_types) {
    for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
        if (stat_types[stat_type]) {
            update_stat_by_stat(device_stats.active[stat_type], pool_stats.active);
            update_stat_by_stat(device_stats.inactive[stat_type], pool_stats.inactive);
            update_stat_by_stat(device_stats.active_bytes[stat_type], pool_stats.active_bytes);
            update_stat_by_stat(device_stats.inactive_bytes[stat_type], pool_stats.inactive_bytes);
        }
    }
}

// -----------------------------------[class HostAllocator]----------------------------------------

template<class T>
StatType HostAllocator::get_stat_type_for_pool(T& pool) {
    if(typeid(T) == typeid(MapBlockPool)) {
        return StatType::FIRST_POOL;
    }else{
        if(static_cast<void*>(&pool) == static_cast<void*>(&second_pool)) {
            return StatType::SECOND_POOL;
        }else{
            return StatType::THIRD_POOL;
        }
    }
}

template<class T>
bool HostAllocator::free_blocks_with_pool(T& pool, int idx, PoolStats& pool_stats) {
    int _try = 0;
    do{
        if(mutex[idx].try_lock()) {
            pool.free_blocks(pool_stats);
            mutex[idx].unlock();
            return true;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }while(_try++ < 5);
    return false;
}

bool HostAllocator::free_cache_blocks() {
    bool flag = false;
    PoolStats pool_stats;
    StatTypes stat_types;
    stat_types[static_cast<int64_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<int64_t>(StatType::FIRST_POOL)] = true;
    stat_types[static_cast<int64_t>(StatType::SECOND_POOL)] = true;
    stat_types[static_cast<int64_t>(StatType::THIRD_POOL)] = true;

    flag = free_blocks_with_pool(first_pool, 0, pool_stats) || flag;
    flag = free_blocks_with_pool(second_pool, 1, pool_stats) || flag;
    flag = free_blocks_with_pool(third_pool, 2, pool_stats) || flag;

    update_stat_array(stats, pool_stats, stat_types);
    return flag;
}

template<class T>
Block* HostAllocator::malloc_with_pool(size_t size, T& pool) {
    PoolType pool_type = pool.pool_type();
    Block* block = new Block(size, pool_type);
    Block* ptr = nullptr;
    StatTypes stat_types;
    stat_types[static_cast<int64_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<int64_t>(get_stat_type_for_pool(pool))] = true;
    PoolStats pool_stats;

    bool block_found = pool.get_free_block(*block, &ptr, pool_stats) 
                    || pool.alloc_block(*block, false, &ptr, pool_stats)
                    || (free_cache_blocks() 
                        && pool.alloc_block(*block, true, &ptr, pool_stats));

    if(!block_found) {
        NNF_CHECK_WITH(nnf::utils::OutOfMemoryError, false,
        "Out of memory. Tried to allocate ", format_size(size), 
        " CPU; ",
        format_size(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " already allocated; ",
        format_size(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current + 
                    stats.inactive_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " reserved in total by NNF)");
    }

    update_stat_array(stats, pool_stats, stat_types);
    return ptr;
}

template<class T>
void HostAllocator::free_with_pool(Block* block, T& pool) {
    PoolStats pool_stats;
    StatTypes stat_types;
    stat_types[static_cast<int64_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<int64_t>(get_stat_type_for_pool(pool))] = true;

    pool.free_block(block, pool_stats);
    update_stat_array(stats, pool_stats, stat_types);
}

Block* HostAllocator::malloc(size_t size) {
    if (size < kMinBlockSize) {
        std::lock_guard<std::recursive_mutex> lock(mutex[0]);
        return malloc_with_pool(size, first_pool);
    } else if (size < kSmallBlockSize) {
        std::lock_guard<std::recursive_mutex> lock(mutex[1]);
        return malloc_with_pool(size, second_pool);
    } else {
        std::lock_guard<std::recursive_mutex> lock(mutex[2]);
        return malloc_with_pool(size, third_pool);
    }
}

void HostAllocator::free(Block* block) {
    size_t size = block->size;
    if (size < kMinBlockSize) {
        std::lock_guard<std::recursive_mutex> lock(mutex[0]);
        free_with_pool(block, first_pool);
    } else if (size < kSmallBlockSize) {
        std::lock_guard<std::recursive_mutex> lock(mutex[1]);
        free_with_pool(block, second_pool);
    } else {
        std::lock_guard<std::recursive_mutex> lock(mutex[2]);
        free_with_pool(block, third_pool);
    }
}

} // namespace cache

} // namespace cpu

} // namespace nnf

#endif