#ifndef NNF_STREAM_H
#define NNF_STREAM_H

#include <src/NNFBase/core/NNFDevice.h>
#include <src/NNFBase/core/NNFDevice.cpp>

namespace nnf {

namespace core {

using uStreamId8 = uint8_t;

class Stream {
protected:
    Device _device;
    uStreamId8 _stream_id = kUndefinedStreamId;
public:
    static const uStreamId8 kUndefinedStreamId = static_cast<uStreamId8>((1<<8) - 1);
    enum Unsafe { UNSAFE };
    enum Default { DEFAULT };

    /*
    Unsafely construct a stream from a Device and a StreamId.  In general, only specific implementations of 
    streams for a backend should manufacture Stream directly in this way; other users should use the provided 
    APIs to get a stream.  In particular, we don't require backends to give any guarantees about non-zero 
    StreamIds; they are welcome to allocate in whatever way they like.
    */
    explicit Stream(Unsafe, Device device, uStreamId8 stream_id) : _device(device), _stream_id(stream_id) {}

    /*
    Construct the default stream of a Device. The default stream is NOT the same as the current stream; default 
    stream is a fixed stream that never changes, whereas the current stream may be changed by StreamGuard.
    */
    explicit Stream(Default, Device device) : _device(device), _stream_id(0) {}

    bool operator==(const Stream& other) const noexcept {
        return this->_device == other.device() && this->_stream_id == other.stream_id();
    }

    bool operator!=(const Stream& other) const noexcept {
        return !(*this == other);
    }


    Device device() const noexcept { return _device; }
    DeviceType device_type() const noexcept { return _device.device_type(); }
    uDeviceIndex8 device_index() const noexcept { return _device.device_index(); }
    uStreamId8 stream_id() const noexcept { return _stream_id; }

    int32_t pack() {
        uint32_t _id = (static_cast<uint32_t>(_device.device_type()) << 16) 
                     | (static_cast<uint32_t>(_device.device_index()) << 8)
                     | (static_cast<uint32_t>(_device.device_type()));
        return _id;
    }

    void set_device_index(uDeviceIndex8 device_index) {
        _device.set_index(device_index);
    }
    void set_device_index_no_check(uDeviceIndex8 device_index) noexcept {
        _device.set_index_no_check(device_index);
    }
    void set_stream_id(uStreamId8 stream_id) noexcept { _stream_id = stream_id;}
    /*
    Enqueues a wait instruction in the stream's work queue. This instruction is a no-op unless the event is 
    marked for recording. In that case the stream stops processing until the event is recorded.
    */
    template <typename T> void wait(const T& event) {
        event.block(*this);
    }

};

std::ostream& operator<<(std::ostream& os, const Stream& ds);

} // namespace core

} // namespace nnf

#endif