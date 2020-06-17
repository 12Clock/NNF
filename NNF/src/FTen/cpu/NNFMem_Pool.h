#ifndef NNF_MEM_POOL_H
#define NNF_MEM_POOL_H

#define NNF_MEM_BLOCK_SIZE 1024

#include <stdlib.h>
#include <unordered_map>

struct NNFMemBlock
{
    size_t size;
    void *ptr;
    bool allocated;
    Block* prev;
    Block* next;
};


class NNFMemPool
{
    private:
        void * pool
}

#endif