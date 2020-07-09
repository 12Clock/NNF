#ifndef NNF_DEVICE_GUARD_IMPL_INTERFACE_H
#define NNF_DEVICE_GUARD_IMPL_INTERFACE_H

#include <src/NNFBase/core/NNFDevice.h>
#include <src/NNFBase/core/NNFDevice.cpp>
#include <src/NNFBase/core/NNFStream.h>
#include <src/NNFBase/core/NNFStream.cpp>
#include <src/NNFBase/NNFBaseMacros.h>

namespace nnf {

namespace core {

enum class EventFlag {
    // CUDA flags
    CUDA_EVENT_DEFAULT,
    CUDA_EVENT_DISABLE_TIMING,
    // FOR TESTING ONLY
    INVALID
};

struct DeviceGuardImplInterface {
    virtual DeviceType device_type() const = 0;
    virtual Device exchangeDevice(Device) const = 0;
    virtual Device getDevice(Device) const = 0;
    virtual void setDevice(Device) const = 0;
    virtual void uncheckedSetDevice(Device) const noexcept = 0;
    virtual DeviceStream getDeviceStream(Device) const noexcept = 0;
    virtual DeviceStream getDefaultDeviceStream(Device) const {
        NNF_CHECK(false, "Backend doesn't support acquiring a default stream.")
    }
    virtual DeviceStream exchangeStream(DeviceStream) const noexcept = 0;


    virtual void destroyEvent(
        void* event, 
        const uDeviceIndex8 device_index) const noexcept {}

    virtual void record(
        void** event,
        const DeviceStream& stream,
        const uDeviceIndex8 device_index,
        const EventFlag flag) const {
        NNF_CHECK(false, "Backend doesn't support events.");
    }
    
    virtual void block(
        void* event,
        const DeviceStream& stream) const {
        NNF_CHECK(false, "Backend doesn't support events.");
    }

    virtual bool queryEvent(
        void* event) const {
        NNF_CHECK(false, "Backend doesn't support events.");
    }

    virtual uDeviceIndex8 deviceCount() const noexcept = 0;
    virtual ~DeviceGuardImplInterface() = default;
};

} // namespace core

} // namespace nnf

#endif