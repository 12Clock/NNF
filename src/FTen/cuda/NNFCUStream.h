#ifndef NNFCU_STREAM_H
#define NNFCU_STREAM_H

#include <src/FTen/cuda/NNFCUMacros.h>
#include <src/NNFBase/core/NNFStream.h>
#include <src/NNFBase/core/NNFStream.cpp>

#include <cuda_runtime_api.h>
#include <mutex>
#include <atomic>
#include <array>
#include <queue>

namespace nnf {

namespace cuda {

using uDeviceIndex8 = nnf::core::uDeviceIndex8;
using DeviceType = nnf::core::DeviceType;
using Stream = nnf::core::Stream;
using Device = nnf::core::Device;
using uStreamId8 = nnf::core::uStreamId8;

/*
Where StreamIdType:
00|xxxxx = default stream
01|xxxxx = low priority stream
10|xxxxx = high priority stream
*/
enum struct StreamIdType: uStreamId8 {
    DEFAULT = 0x00,
    LOW = 0x40,
    HIGH = 0x80
};

/*
Non-blocking streams which do not synchronize with the legacy stream can be 
created using the cudaStreamNonBlocking flag with the stream creation APIs.
*/

using StreamIdIndex = uStreamId8;

class CUDAStream final : public Stream
{
private:
    cudaStream_t _cuda_stream = nullptr;
public:
    /*
    Note [StreamId assignment]
    |=========================\
    How do we assign stream IDs?
    -- 1 bits -- | -- 2 bits -- | -- 5 bit --
    zeros        | StreamIdType   | stream id index
    */
    static constexpr uStreamId8 kStreamIdTypeMask = 0x60; // mask of StreamIdType
    static constexpr uStreamId8 kStreamIdIndexMask = 0x1f; // mask of StreamIdIndex

    explicit CUDAStream(uDeviceIndex8 device_index, uStreamId8 stream_id) 
    : Stream(Stream::UNSAFE, Device(DeviceType::CUDA, device_index), stream_id) {}

    explicit CUDAStream(uDeviceIndex8 device_index) 
    : Stream(Stream::DEFAULT, Device(DeviceType::CUDA, device_index)) {}

    explicit CUDAStream(uDeviceIndex8 device_index, StreamIdIndex stream_id_index, StreamIdType stream_id_type) 
    : Stream(Stream::UNSAFE, 
             Device(DeviceType::CUDA, 
             device_index)
    , static_cast<uStreamId8>(stream_id_type) | stream_id_index) {}

    bool operator==(const CUDAStream& other) const noexcept {
        return this->_device == other.device() && this->_stream_id == other.stream_id();
    }

    bool operator!=(const CUDAStream& other) const noexcept {
        return !(*this == other);
    }

    cudaStream_t cudaStream() const noexcept { return _cuda_stream;}
    StreamIdIndex stream_id_index() const noexcept { return _stream_id & kStreamIdIndexMask;}
    StreamIdType stream_id_type() const noexcept { 
        return static_cast<StreamIdType>(_stream_id & kStreamIdTypeMask);
    }

    void set_cudaStream(cudaStream_t cuda_stream) noexcept { _cuda_stream = cuda_stream;}
    void set_stream_id_index(StreamIdIndex stream_id_index) { 
        _stream_id = (_stream_id & kStreamIdTypeMask) | stream_id_index;
    }
    void set_stream_id_type(StreamIdType stream_id_type) { 
        _stream_id = (_stream_id & kStreamIdIndexMask) | static_cast<uStreamId8>(stream_id_type);
    }

};

} // namespace cuda

} // namespace nnf

namespace std {
    template <>
    struct hash<nnf::cuda::CUDAStream> {
            size_t operator()(nnf::cuda::CUDAStream s) const noexcept {
            return s.pack();
        }
    };
} // namespace std


#endif