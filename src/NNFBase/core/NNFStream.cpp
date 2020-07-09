#ifndef NNF_STREAM_CPP
#define NNF_STREAM_CPP

#include <src/NNFBase/core/NNFStream.h>

namespace nnf {

namespace core {

std::ostream& operator<<(std::ostream& os, Stream& ds) {
    os << "stream " << ds.stream_id() << " on device " << ds.device().str();
    return os;
}

} // namespace nnf

} // namespace core

#endif