#include <iostream>
#include <src/FTen/NNFAllocator.h>
#include <thread>

using namespace std;

using host = nnf::cpu::cache::Block;

void get_ptr(size_t size, host* ptr) 
{
    ptr = nnf::cache::NNFAllocator(host).malloc(size);
    nnf::cache::NNFAllocator(host).free(ptr);
}

int main()
{
    host* ptr;
    thread t1(get_ptr, 4, ptr);
    t1.join();

    // device* ptr = allocator.malloc<device>(4);
    // allocator.free<device>(ptr);

    cout<<"SUCCESS"<<endl;
    return 0;
}