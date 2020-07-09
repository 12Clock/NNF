#ifndef NNFC_MEMORY_POOL_H
#define NNFC_MEMORY_POOL_H

#include <src/FTen/cpu/NNFCMacros.h>
#include <unordered_map>
#include <set>
#include <deque>
#include <stack>

namespace nnf {

namespace cpu {

namespace cache {

constexpr size_t kMinBlockSize = 4096;        // Less than 4KiB memory belong first-level pool
constexpr size_t kFirstPoolSize = 8;          // First-level allocation is 8B memory alignment
constexpr size_t kMinBlockBuffer = 8;         // First-level allocable memery times
constexpr size_t kSmallBlockSize = 2097152;   // Less than 2MiB memory belong second-level pool
constexpr size_t kSecondBlockSize = 1048576;  // Second-level allocation is 1MiB memory alignment
constexpr size_t kLargeBlockSize = 20971520;  // Larger than 20MiB don't use memory alignment
constexpr size_t kThirdBlockSize = 1048576;   // Third-level allocation is 1MiB memory alignment

enum struct PoolType {
    UNDEFINED = 0,
    FIRST     = 1,
    SECOND    = 2,
    THIRD     = 3
};

struct Block;
typedef std::stack<Block*> StackPool;

struct Block {
    StackPool* _it;               // 
    size_t    size;               // block size in bytes
    PoolType  pool_type;          // Level of pool
    void*     ptr;                // Memory address
    bool      allocated;          // In-use flag   

    Block*    prev;               // Prev block if split from a larger allocation
    Block*    next;               // Next block if split from a larger allocation
    
    Block() {}

    Block(StackPool* it, size_t size, PoolType pool, void *ptr)
    : _it(it), size(size), pool_type(pool), ptr(ptr), 
    allocated(false), prev(nullptr), next(nullptr) {}

    Block(size_t size, PoolType pool, void *ptr)
    : _it(nullptr), size(size), pool_type(pool), ptr(ptr), 
    allocated(false), prev(nullptr), next(nullptr) {}

    Block(size_t size, PoolType pool)
    : _it(nullptr), size(size), pool_type(pool), ptr(nullptr), 
    allocated(false), prev(nullptr), next(nullptr) {}

    bool is_split() const {
        return (prev != nullptr) || (next != nullptr);
    }

    bool operator<(const Block* other) const
    {
        if (size != other->size) {
            return size < other->size;
        }
        return (uintptr_t)ptr < (uintptr_t)other->ptr;
    }
};

struct Stat {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

void reset_accumulated_stat(Stat& stat);
void reset_peak_stat(Stat& stat);
void update_stat(Stat& stat, int64_t amount);

struct PoolStats {
    Stat active;                // COUNT: number of active memory blocks (allocated)
    Stat inactive;              // COUNT: number of inactive, split memory blocks (unallocated but can't be released via cudaFree)

    Stat active_bytes;          // SUM: bytes within active memory blocks
    Stat inactive_bytes;        // SUM: bytes within inactive, split memory blocks
};


static size_t round_size(const size_t size);

class MapBlockPool {
private:
    std::unordered_map<size_t, std::deque<StackPool>> free_pool;
    PoolType _pool_type;
public:
    MapBlockPool() : _pool_type(PoolType::UNDEFINED) {}
    MapBlockPool(PoolType pool_type) : _pool_type(pool_type) {}
    bool get_free_block(Block& block, Block** ptr, PoolStats& pool_stats);
    void free_block(Block* block, PoolStats& pool_stats);
    bool alloc_block(Block& block, bool isRetry, Block** ptr, PoolStats& pool_stats);
    void free_blocks(PoolStats& pool_stats);
    
    PoolType pool_type() const { return _pool_type; }
};

class SetBlockPool {
private:
    std::set<Block*> free_pool;
    PoolType _pool_type;
    size_t _pool_infimum = 0;

    void try_merge_blocks(Block* dst, Block* src, PoolStats& pool_stats);
public:
    SetBlockPool() : _pool_type(PoolType::UNDEFINED) {}
    SetBlockPool(PoolType pool_type) : _pool_type(pool_type) {
        switch (pool_type)
        {
            case PoolType::FIRST:
                break;
            case PoolType::SECOND:
                _pool_infimum = kMinBlockSize;
                break;
            case PoolType::THIRD:
                _pool_infimum = kSmallBlockSize;
                break;
            default:
                break;
        }
    }
    bool get_free_block(Block& block, Block** ptr, PoolStats& pool_stats);
    void free_block(Block* block, PoolStats& pool_stats);
    bool alloc_block(Block& block, bool isRetry, Block** ptr, PoolStats& pool_stats);
    void free_blocks(PoolStats& pool_stats);

    PoolType pool_type() const { return _pool_type; }
};

} // namespace cache

} // namespace cpu

} // namespace nnf

#endif