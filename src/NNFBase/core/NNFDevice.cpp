#ifndef NNF_DEVICE_CPP
#define NNF_DEVICE_CPP

#include <src/NNFBase/core/NNFDevice.h>
#include <algorithm>
#include <array>
#include <regex>

namespace nnf {

namespace core {

DeviceType parse_type(const std::string& device_string){
    static const std::array<std::pair<std::string, DeviceType>, 2> mDeviceTypes = {{
        {"cpu",  DeviceType::CPU},
        {"cuda", DeviceType::CUDA}
    }};
    auto device = std::find_if(
        mDeviceTypes.begin(),
        mDeviceTypes.end(),
        [device_string](const std::pair<std::string, DeviceType>& p) {
            return p.first == device_string;
        });
    if (device != mDeviceTypes.end()) {
        return device->second;
    }
    NNF_ERROR("Expected one of cpu, cuda device type at start of device string: '", device_string, "'");
}

Device::Device(const std::string& device_string) : Device(DeviceType::CPU) {
    NNF_CHECK(!device_string.empty(), "Device string must not be empty");

    // We assume gcc 5+, so we can use proper regex.
    static const std::regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
    std::smatch sm;
    NNF_CHECK(std::regex_match(device_string, sm, regex), 
        "Invalid device string: '", device_string, "'");
    _device_type = parse_type(sm[1].str());
    if(sm[2].matched){
        try{
            _device_index = nnf::details::stoi(sm[2].str());
        } catch (const std::exception &) {
            NNF_ERROR(
                "Could not parse device index '", sm[2].str(),
                "' in device string '", device_string, "'");
        }
    }
    val_device_index();
    val_device_type();
}

std::string Device::str() const {
    std::string str = DeviceTypeToString(_device_type);
    if(has_index()) {
        str.push_back(':');
        str.append(std::to_string(_device_index));
    }
    return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << device.str();
    return stream;
}

} // namespace core

} // namespace nnf

#endif