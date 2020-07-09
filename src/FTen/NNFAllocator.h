#ifndef NNF_ALLOCATOR_H
#define NNF_ALLOCATOR_H

#include <src/FTen/cpu/NNFCAllocator.h>
#include <src/FTen/cpu/NNFCAllocator.cpp>
#include <src/FTen/cuda/NNFCUAllocator.h>
#include <src/FTen/cuda/NNFCUAllocator.cpp>

namespace nnf {

namespace cache {

using HostAllocator = nnf::cpu::cache::HostAllocator;
using DeviceAllocator = nnf::cuda::cache::DeviceAllocator;

HostAllocator host_allocator;
DeviceAllocator device_allocator;

#define NNFAllocator(type) type##_allocator

} // namespace cache

} // namespace nnf

#endif