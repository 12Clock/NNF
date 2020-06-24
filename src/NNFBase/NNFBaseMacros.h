#ifndef NNF_BASE_MACROS_H
#define NNF_BASE_MACROS_H

// The base macros of NNF

#include <src/NNFBase/utils/NNFExceptionMacros.h>

#define NNF2STRING(x) #x
#define NNF_CONCAT2(x, y) x##y
#define NNF_CONCAT3(x, y, z) x##y##z
#define NNF_CONCAT4(x, y, z, a) x##y##z##a

#define NNF_STR_CONCAT2(x, y) (static_cast<std::string>(x)+y)
#define NNF_STR_CONCAT3(x, y, z) (static_cast<std::string>(x)+y+z)
#define NNF_STR_CONCAT4(x, y, z, a) (static_cast<std::string>(x)+y+z+a)
#define NNF_STR_CONCAT5(a, b, c, d, e) (static_cast<std::string>(a)+b+c+d+e)

#endif