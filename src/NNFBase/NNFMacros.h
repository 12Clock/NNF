#ifndef NNF_MACROS_H
#define NNF_MACROS_H

#define NNF2STRING(x) #x
#define NNF_CONCAT2(x, y) x##y
#define NNF_CONCAT3(x, y, z) x##y##z
#define NNF_CONCAT4(x, y, z, a) x##y##z##a

#include <string>
#define NNF_STR_CONCAT2(x, y) (static_cast<std::string>(x)+y)
#define NNF_STR_CONCAT3(x, y, z) (static_cast<std::string>(x)+y+z)
#define NNF_STR_CONCAT4(x, y, z, a) (static_cast<std::string>(x)+y+z+a)
#define NNF_STR_CONCAT5(a, b, c, d, e) (static_cast<std::string>(a)+b+c+d+e)

#include <iostream>
#define NNF_ERROR_PRINT(x) std::cerr<<x<<std::endl;

#define NNF_CUDA_CHECK(ERR, FUN, FILE)                                  \
do{                                                                     \
if(ERR != cudaSuccess){                                                 \
NNF_ERROR_PRINT(NNF_STR_CONCAT5(                                        \
    cudaGetErrorString(ERR), "(", FUN, ") in Files: ", FILE             \
));                                                                     \
}}while(0)

#endif