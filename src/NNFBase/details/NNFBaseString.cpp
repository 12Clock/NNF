#ifndef NNF_BASE_STRING_CPP
#define NNF_BASE_STRING_CPP

#include <sstream>

namespace nnf{

namespace details
{
    template<typename ... Args>
    inline decltype(auto) str(const Args&... args)
    {
        std::stringstream oss;
        for(auto p: {args...}){
            oss << p;
        }
        return oss.str();
    }
} // namespace details

} // namespace nnf

#endif