#ifndef NNFC_MEMORY_POOL_CPP
#define NNFC_MEMORY_POOL_CPP

#include <src/FTen/cpu/NNFCMemoryPool.h>

namespace nnf {

namespace cpu {

namespace cache {

static size_t round_size(const size_t size)
{
    if(size < kMinBlockSize) {
        return (size + kFirstPoolSize - 1) / kFirstPoolSize * kFirstPoolSize;
    }else if(size < kSmallBlockSize) {
        return (size + kSecondBlockSize - 1) / kSecondBlockSize * kSecondBlockSize;
    }else if (size < kLargeBlockSize) {
        return (size + kThirdBlockSize - 1) / kThirdBlockSize * kThirdBlockSize;
    }else{
        return size;
    }
}

//-----------------------------------[struct Stat]----------------------------------------

void reset_accumulated_stat(Stat& stat) {
    stat.allocated = 0;
    stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
    stat.peak = stat.current;
}

void update_stat(Stat& stat, int64_t amount) {
    stat.current += amount;

    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }

    if (amount < 0) {
        stat.freed += -amount;
    }
}

//------------------------------------[class MapBlockPool]------------------------------------------
bool MapBlockPool::get_free_block(Block& block, Block** ptr, PoolStats& pool_stats) {
    size_t size = round_size(block.size);
    if(free_pool.find(size) == free_pool.end()) return false;
    for(int i = 0; i < free_pool[size].size(); ++i) {
        StackPool stack_pool = free_pool[size].front();
        if(stack_pool.empty()) {
            free_pool[size].pop_front();
            free_pool[size].emplace_back(stack_pool);
            continue;
        }
        *ptr = stack_pool.top();
        (*ptr)->allocated = true;
        stack_pool.pop();
        update_stat(pool_stats.active, 1);
        update_stat(pool_stats.inactive, -1);
        update_stat(pool_stats.active_bytes, size);
        update_stat(pool_stats.inactive_bytes, size);
        return true;
    }
    return false;
}

void MapBlockPool::free_block(Block* block, PoolStats& pool_stats) {
    NNF_CHECK(block->allocated, "Can't free unallocated block!");
    block->_it->emplace(block);
    update_stat(pool_stats.active, -1);
    update_stat(pool_stats.inactive, 1);
    update_stat(pool_stats.active_bytes, -(block->size));
    update_stat(pool_stats.inactive_bytes, block->size);
}

bool MapBlockPool::alloc_block(Block& block, 
                               bool isRetry, 
                               Block** ptr, 
                               PoolStats& pool_stats) {
    const size_t size = round_size(block.size);
    char* _ptr = (char*)malloc(kMinBlockBuffer * size);
    if(_ptr == nullptr) return false;
    free_pool[size].emplace_back(); 
    StackPool& _block_stack = free_pool[size].back();
    Block* tmp = new Block(&_block_stack, size, PoolType::FIRST, static_cast<void*>(_ptr));
    for(int i = 1; i < kMinBlockBuffer; ++i) {
        tmp->next = new Block(&_block_stack, size, PoolType::FIRST, static_cast<void*>(_ptr + i * size));
        tmp->next->prev = tmp;
        _block_stack.emplace(tmp);
        tmp = tmp->next;
    }
    tmp->allocated = true;
    *ptr = tmp;
    update_stat(pool_stats.active, 1);
    update_stat(pool_stats.inactive, kMinBlockBuffer - 1);
    update_stat(pool_stats.active_bytes, size);
    update_stat(pool_stats.inactive_bytes, size * (kMinBlockBuffer - 1));
    return true;
}

void MapBlockPool::free_blocks(PoolStats& pool_stats) {
    for(decltype(free_pool.begin()) i = free_pool.begin(); i != free_pool.end(); ++i) {
        size_t size = i->first;
        auto tq = i->second;
        for(decltype(tq.begin()) tq_it = tq.begin(); tq_it != tq.end(); ++tq_it) {
            if(tq_it->size() == kMinBlockBuffer) {
                auto ptr = tq_it->top();
                if(ptr->next) {
                    auto next = ptr->next;
                    while(next->next) {
                        auto cur = next;
                        next = next->next;
                        delete cur;
                    }
                }
                while(ptr->next) {
                    auto cur = ptr;
                    ptr = ptr->prev;
                    delete cur;
                };
                tq.erase(tq_it);
                free(ptr);
                delete ptr;

                update_stat(pool_stats.inactive, -kMinBlockBuffer);
                update_stat(pool_stats.inactive_bytes, -size * kMinBlockBuffer);
            }
        }
    }
}

//------------------------------------[class SetBlockPool]------------------------------------------

void SetBlockPool::try_merge_blocks(Block* dst, Block* src, PoolStats& pool_stats) {
    if (!src || src->allocated) return;

    NNF_CHECK(dst->is_split() && src->is_split(), "Error merging blocks");

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    free_pool.erase(src);
    delete src;

    update_stat(pool_stats.inactive, -1);
}

bool SetBlockPool::get_free_block(Block& block, Block** ptr, PoolStats& pool_stats) {
    auto it = free_pool.lower_bound(&block);
    if (it == free_pool.end()) return false;
    *ptr = *it;
    free_pool.erase(it);
    size_t size = block.size;
    if((*ptr)->size - size > _pool_infimum){
        Block* remaining = new Block((*ptr)->size - size, block.pool_type);
        remaining->ptr = static_cast<void*>(static_cast<char*>((*ptr)->ptr) + size);
        if((*ptr)->next) (*ptr)->next->prev = remaining;
        remaining->next = (*ptr)->next;
        (*ptr)->next = remaining;
        remaining->prev = (*ptr);
        free_pool.insert(remaining);
        update_stat(pool_stats.inactive, 1);
        update_stat(pool_stats.inactive_bytes, (remaining->size));
    }
    update_stat(pool_stats.active, 1);
    update_stat(pool_stats.inactive, -1);
    update_stat(pool_stats.active_bytes, (*ptr)->size);
    update_stat(pool_stats.inactive_bytes, -((*ptr)->size));
    return true;
}

void SetBlockPool::free_block(Block* block, PoolStats& pool_stats) {
    NNF_CHECK(block->allocated, "Can't free unallocated block!");

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
        try_merge_blocks(block, merge_candidate, pool_stats);
    }

    free_pool.insert(block);
    update_stat(pool_stats.active, -1);
    update_stat(pool_stats.inactive, 1);
    update_stat(pool_stats.active_bytes, -(block->size));
    update_stat(pool_stats.inactive_bytes, block->size);
}

bool SetBlockPool::alloc_block(Block& block, 
                               bool isRetry, 
                               Block** ptr, 
                               PoolStats& pool_stats) {
    size_t size = block.size;
    void* _ptr = (void*)malloc(size);
    if(_ptr == nullptr) return false;
    (*ptr)->ptr = _ptr;

    update_stat(pool_stats.active, 1);
    update_stat(pool_stats.active_bytes, size);
    return true;
}

void SetBlockPool::free_blocks(PoolStats& pool_stats) {
    auto it = free_pool.begin();
    while (it != free_pool.end()) {
        Block* block = *it;
        if (!block->prev && !block->next) {
            free(block->ptr);
            
            auto cur = it;
            ++it;
            free_pool.erase(cur);
            delete block;

            update_stat(pool_stats.inactive, -1);
            update_stat(pool_stats.inactive_bytes, -(block->size));
        } else {
            ++it;
        }
    }
}

} // namespace cache

} // namespace cpu

} // namespace nnf

#endif