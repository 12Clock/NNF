#ifndef NNF_DEVICE_H
#define NNF_DEVICE_H

#include <src/NNFBase/NNFBaseMacros.h>

namespace nnf {

namespace core {

using uDeviceIndex8 = uint8_t;

enum struct DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    UNDEFINED = static_cast<uint8_t>((1<<8)-1)
};

std::string DeviceTypeToString(DeviceType device_type)
{
    switch (device_type)
    {
    case DeviceType::CPU:
        return "cpu";
    case DeviceType::CUDA:
        return "cuda";
    default:
        NNF_ERROR("Can't transform invalid device type to string, device type : ", device_type);
    }
}

struct Device final {
public:
    const uDeviceIndex8 kUndefinedDeviceIndex = static_cast<uDeviceIndex8>((1<<8)-1);
private:
    uDeviceIndex8 _device_index = kUndefinedDeviceIndex;
    DeviceType _device_type = DeviceType::UNDEFINED;
    void val_device_type() const
    {
        switch(_device_type){
            case DeviceType::CPU:
            case DeviceType::CUDA:
                return;
            default:
                NNF_ERROR("Device type is not in cpu, cuda. ( device_type : ", _device_type, ")");
        }
    }
    void val_device_index() const
    {
        NNF_CHECK(!is_cpu() || _device_index != kUndefinedDeviceIndex,
            "CPU device index must be 0 or ", kUndefinedDeviceIndex, "(Undefined), but got ", _device_index);
    }
public:
    Device(DeviceType device_type) : 
    _device_type(device_type) {
        val_device_index();
        val_device_type();
    }

    Device(DeviceType device_type, uDeviceIndex8 device_index) : 
    _device_type(device_type), 
    _device_index(device_index) {
        val_device_index();
        val_device_type();
    }

    Device(const std::string& device_string);

    bool operator==(const Device& other) const noexcept {
        return this->_device_type == other.device_type() && 
            this->_device_index == other.device_index();
    }

    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

    /// Sets the device index, no check.
    void set_index_no_check(uDeviceIndex8 index) noexcept{
        _device_index = index;
    }

    void set_index(uDeviceIndex8 index) {
        _device_index = index;
        val_device_index();
    }

    DeviceType device_type() const noexcept {
        return _device_type;
    }

   uDeviceIndex8 device_index() const noexcept {
        return _device_index;
    }

    bool is_cpu() const noexcept {
        return _device_type == DeviceType::CPU;
    }

    bool is_cuda() const noexcept {
        return _device_type == DeviceType::CUDA;
    }

    bool has_index() const noexcept {
        return _device_index != -1;
    }

    std::string str() const;
};

} // namespace core

} // namespace nnf

#endif