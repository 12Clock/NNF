#ifndef NNFCU_ALLOCATOR_CPP
#define NNFCU_ALLOCATOR_CPP

#include <src/FTen/cuda/NNFCUMacros.h>
#include <src/FTen/cuda/NNFCUAllocator.h>
#include <assert.h>
#include <sstream>
#include <iostream>
#include <cuda_runtime_api.h>

namespace nnf {

namespace cuda {

namespace cache {

// ============================[Stat, DeviceStats function]================================
void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  NNF_INTERNAL_ASSERT(stat.current >= 0, "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

void update_stat_array(StatArray& stat_array, int64_t amount, const StatTypes& stat_types) {
  for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

// ==============================[DeviceAllocator functions]====================================

size_t DeviceAllocator::try_merge_blocks(CUDABlock* dst, CUDABlock* src, CUDABlockPool& pool)
{
    if(!src || src->allocated || src->event_count > 0){
        return 0;
    }

    if(dst->prev == src){ 
        // In the cuda memory pool, the dst block is in front of the src block
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if(dst->prev){
            dst->prev->next = dst;
        }
    }else{
        // In the cuda memory pool, the dst block is in back of the src block
        dst->next = src->next;
        if(dst->next){
            dst->next->prev = dst;
        }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    pool.erase(src);
    delete src;

    return subsumed_size;
}

StatType DeviceAllocator::get_stat_type_for_pool(const CUDABlockPool& pool) const
{
    if(&pool == &small_blocks){
        return StatType::SMALL_POOL;
    }else if(&pool == &large_blocks){
        return StatType::LARGE_POOL;
    }else{
        NNF_ERROR("Can't find type of pool, this pool is Invalid");
    }
}

void DeviceAllocator::free_block(CUDABlock* block)
{
    if(!block->allocated && block->event_count == 0){
        NNF_ERROR("ERROR: Can't free unallocated block and number of events is 0.");
    }

    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<CUDABlock*, 2> merge_candidates = {block->prev, block->next};
    for (CUDABlock* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    pool.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
    update_stat_array(stats.active, -1, stat_types);
    update_stat_array(stats.active_bytes, -original_block_size, stat_types);
}

void DeviceAllocator::process_events()
{
    while (!cuda_events.empty())
    {
        auto &e = cuda_events.front();
        cudaEvent_t event = e.first;
        CUDABlock *block = e.second;

        cudaError_t err = cudaEventQuery(event);
        if(err == cudaErrorNotReady){
            /*
            cudaErrorNotReady:
            This indicates that asynchronous operations issued previously 
            have not completed yet. This result is not actually an error, 
            but must be indicated differently than cudaSuccess (which indicates 
            completion). Calls that may return this value include cudaEventQuery() 
            and cudaStreamQuery().
            */
            cudaGetLastError(); // make cudaErrorNotReady to cudaSuccess
            break;
        }else if(err != cudaSuccess){
            NNF_ERROR(cudaGetErrorString(err));
        }

        err = cudaEventDestroy(event); //Destroy event that run successfully
        if(err != cudaSuccess){
            NNF_ERROR(cudaGetErrorString(err));
        }

        block->event_count--;
        if(block->event_count == 0){
            free_block(block);
        }

        cuda_events.pop_front();
    }
}

CUDABlockPool& DeviceAllocator::get_pool(size_t size)
{
    if(size <= kSmallSize){
        return small_blocks;
    }else{
        return large_blocks;
    }
}

size_t DeviceAllocator::get_allocation_size(const size_t size)
{
    if(size <= kSmallSize){
        return kSmallBuffer;
    }else if(size < kMinLargeAlloc){
        return kLargeBuffer;
    }else{
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

bool DeviceAllocator::get_free_block(AllocParams& params)
{
    CUDABlockPool& pool = *params.pool;
    auto it = pool.lower_bound(&params.search_key);
    if (it == pool.end() || (*it)->stream != params.stream()){
        return false;
    }
    params.block = *it;
    pool.erase(it);
    return true;
}

bool DeviceAllocator::trigger_free_memory_callbacks(AllocParams& params)
{
    // Skip this part, there are no GC
    bool freed_memory = false;
    return freed_memory;
}

bool DeviceAllocator::alloc_block(AllocParams& params, bool isRetry)
{
    size_t size = params.alloc_size;
    void* ptr;

    if(isRetry){
        stats.num_alloc_retries += 1;
    }

    params.err = cudaMalloc(&ptr, size);
    if(params.err != cudaSuccess){
        if(isRetry || params.err == cudaErrorMemoryAllocation){
            cudaGetLastError();
        }
        return false;
    }

    params.block = new CUDABlock(params.device(), params.stream(), size, params.pool, ptr);
    update_stat_array(stats.segment, 1, params.stat_types);
    update_stat_array(stats.reserved_bytes, size, params.stat_types);

    return (params.block != nullptr);
}

void DeviceAllocator::synchronize_and_free_events()
{
    for(auto& e: cuda_events) {
        cudaEvent_t event = e.first;
        CUDABlock* block = e.second;

        NNFCU_CHECK(cudaEventSynchronize(event));
        NNFCU_CHECK(cudaEventDestroy(event));

        block->event_count -= 1;
        if(block->event_count == 0){
            free_block(block);
        }
    }

    cuda_events.clear();
}


void DeviceAllocator::free_blocks(CUDABlockPool& pool)
{
    auto it = pool.begin();
    while(it != pool.end()){
        CUDABlock* block = *it;
        if(!block->prev && !block->next){
            NNFCU_CHECK(cudaFree(block->ptr));

            StatTypes stat_types;
            stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
            stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
            update_stat_array(stats.segment, -1, stat_types);
            update_stat_array(stats.reserved_bytes, -(block->size), stat_types);
            auto cur = it;
            ++it;
            pool.erase(cur);
            delete block;
        }else{
            ++it;
        }
    }
}

bool DeviceAllocator::free_cached_blocks()
{
    synchronize_and_free_events();

    free_blocks(small_blocks);
    free_blocks(large_blocks);
    return true;
}

std::string DeviceAllocator::format_size(size_t size){
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

bool DeviceAllocator::should_split(CUDABlock* block, size_t size)
{
    size_t remaining = block->size - size;
    if(block->pool == &small_blocks){
        return remaining >= kMinBlockSize;
    }else if(block->pool == &large_blocks){
        return remaining >= kSmallSize;
    }else{
        NNF_ERROR("Invalid pool.");
    }
}

cudaEvent_t DeviceAllocator::create_event_internal() {
    cudaEvent_t event;
    NNFCU_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void DeviceAllocator::insert_events(CUDABlock* block)
{
    int prev_device;
    NNFCU_CHECK(cudaGetDevice(&prev_device));
    
    stream_set streams(std::move(block->stream_uses));
    for(auto it = streams.begin(); it != streams.end(); it++){
        NNFCU_CHECK(cudaSetDevice(it->device_index()));

        cudaEvent_t event = create_event_internal();
        NNFCU_CHECK(cudaEventRecord(event, it->cudaStream()));

        block->event_count++;
        cuda_events.emplace_back(event, block);
    }

    NNFCU_CHECK(cudaSetDevice(prev_device));
}

//public:
CUDABlock* DeviceAllocator::malloc(int device, size_t size, cudaStream_t stream)
{
    std::unique_lock<std::recursive_mutex> lock(mutex);

    process_events();

    size = round_size(size);
    auto& pool = get_pool(size);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    bool block_found = 
        get_free_block(params) //Found the appropriate block
        || (trigger_free_memory_callbacks(params) && get_free_block(params))
        || alloc_block(params, false)
        || (free_cached_blocks() && alloc_block(params, true));
    
    NNF_INTERNAL_ASSERT((!block_found && params.err != cudaSuccess || params.block));

    if(!block_found){
        if(params.err == cudaErrorMemoryAllocation){
            size_t device_free;
            size_t device_total;
            cudaMemGetInfo(&device_free, &device_total);
            NNF_CHECK_WITH(nnf::utils::CUDAOutOfMemoryError, false,
                "CUDA out of memory. Tried to allocate ", format_size(alloc_size), 
                " GPU ", NNF2STRING(device), "; ", format_size(device_total), "total capacity; ",
                format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                " already allocated; ", format_size(device_free), " free; ",
                format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                " reserved in total by NNF)");
        } else {
            NNFCU_CHECK(params.err);
        }
    }

    CUDABlock* block = params.block;
    CUDABlock* remaining = nullptr;
    NNF_INTERNAL_ASSERT(block);

    const bool already_spilt = block->is_split();
    if(should_split(block, size)){
        remaining = block;

        block = new CUDABlock(device, stream, size, &pool, block->ptr);
        block->prev = remaining->prev;
        if(block->prev){
            block->prev->next = block;
        }
        block->next = remaining;
        remaining->prev = block;

        remaining->ptr = static_cast<void*>(static_cast<char*>(remaining->ptr) + size);
        remaining->size -= size;
        pool.insert(remaining);

        if(already_spilt){
            // An already-split inactive block is being shrunk by size bytes.
            update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
        }else{
            // A new split inactive block is being created from a previously unsplit block,
            // size remaining->size bytes.
            update_stat_array(stats.inactive_split_bytes, remaining->size, params.stat_types);
            update_stat_array(stats.inactive_split, 1, params.stat_types);
        }
    } else if(already_spilt){
        update_stat_array(stats.inactive_split, -1, params.stat_types);
        update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
    }

    block->allocated = true;
    active_blocks.insert(block);

    update_stat_array(stats.active, 1, params.stat_types);
    update_stat_array(stats.active_bytes, block->size, params.stat_types);
    update_stat_array(stats.allocation, 1, params.stat_types);
    update_stat_array(stats.allocated_bytes, block->size, params.stat_types);

    return block;
}

void DeviceAllocator::free(CUDABlock* block)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.allocation, -1, stat_types);
    update_stat_array(stats.allocated_bytes, -block->size, stat_types);

    if(!block->stream_uses.empty()){
        insert_events(block);
    }else{
        free_block(block);
    }
}

} // namespace cache

} // namespace cuda

} // namespace nnf

#endif