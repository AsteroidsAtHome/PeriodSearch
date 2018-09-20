#include <CL/cl.hpp>
#include <string>
#include <fstream>

cl::Program CreateProgram(const std::string& file)
{
    std::vector<cl::Platform> _platforms;
    cl::Platform::get(&_platforms);

    auto _platform = _platforms.front();
    std::vector<cl::Device> _devices;
    auto err = 0;
    err = _platform.getDevices(CL_DEVICE_TYPE_GPU, &_devices);

    auto _device = _devices.front();
    auto vendor = _device.getInfo<CL_DEVICE_VENDOR>();
    auto version = _device.getInfo<CL_DEVICE_VERSION>();
    auto name = _device.getInfo<CL_DEVICE_NAME>();

    std::ifstream helloWorldFile(std::string("../../period_search/") + file);
    std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));
    const cl::Program::Sources _sources(1, std::make_pair(src.c_str(), src.length() + 1));
    const cl::Context _context(_device);
    cl::Program _program(_context, _sources);

    //err = _program.build(_devices);
    err = _program.build("-cl-std=CL1.2");

    return _program;
}