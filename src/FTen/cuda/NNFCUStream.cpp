#ifndef NNFCU_STREAM_CPP
#define NNFCU_STREAM_CPP

#include <src/FTen/cuda/NNFCUStream.h>

namespace nnf {

namespace cuda {

//--------------------------------[class CUDAStream]--------------------------------------
std::ostream& operator<<(std::ostream& os, StreamIdType stream_id_type){
    switch (stream_id_type)
    {
    case StreamIdType::DEFAULT:
        os << "DEFAULT";
        break;
    case StreamIdType::LOW:
        os << "LOW";
        break;
    case StreamIdType::HIGH:
        os << "HIGH";
        break;
    default:
        os << static_cast<uStreamId8>(stream_id_type);
        break;
    }
    return os;
}

} // namespace cuda

} // namespace nnf

#endif