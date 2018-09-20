#pragma once
#include <string>
#include <CL/cl.hpp>

cl::Program CreateProgram(const std::string& file);
