#pragma once
#include <cuda_runtime_api.h>
class Cc
{
	int deviceCcMajor;
	int deviceCcMinor;

	int GetSmxBlockCuda12() const;
	int GetSmxBlockCuda11() const;
	int GetSmxBlockCuda10() const;
	int GetSmxBlockCuda6() const;
	int GetSmxBlockCc12() const;
	int GetSmxBlockCc9() const;
	int GetSmxBlockCc8() const;
	int GetSmxBlockCc7() const;
	int GetSmxBlockCc6() const;
	int GetSmxBlockCc5() const;
	int GetSmxBlockCc3() const;
	int GetSmxBlockCc2() const;
	int GetSmxBlockCc1() const;
	static const int DefaultSmxBlock = 16; // Safe fallback for unsupported architectures
	int UseDefault() const;

public:

	int cudaVersion;

	explicit Cc(const cudaDeviceProp& deviceProp);
	int GetSmxBlock() const;

#if defined (_MSC_VER) & (_MSC_VER >= 1900) // Visual Studio 2013 or later
	~Cc() = default;
#else
	~Cc();
#endif
	
};
