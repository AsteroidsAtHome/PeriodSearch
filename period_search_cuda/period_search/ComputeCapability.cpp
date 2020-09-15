#include <cstdio>
#include <cstdlib>
#include "ComputeCapability.h"
#include <cuda_runtime_api.h>

Cc::Cc(const cudaDeviceProp deviceProp)
{
	this->cudaVersion = CUDART_VERSION;
	deviceCcMajor = deviceProp.major;
	deviceCcMinor = deviceProp.minor;
}

Cc::~Cc() = default;

int Cc::GetSmxBlock() const
{
	auto result = 0;
	if (cudaVersion >= 11000 && cudaVersion < 12000)
	{
		result = GetSmxBlockCuda11();
	}
	else if (cudaVersion >= 10000 && cudaVersion < 11000)
	{
		result = GetSmxBlockCuda10();
	}

	return result;
}

int Cc::GetSmxBlockCuda11() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 8:
		smxBlock = GetSmxBlockCc8();
		break;
	case 7:
		smxBlock = GetSmxBlockCc7();
		break;
	case 6:
		smxBlock = GetSmxBlockCc6();
		break;
	case 5:
		smxBlock = GetSmxBlockCc5();
		break;
	default:
		Exit();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCuda10() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 8:
		smxBlock = GetSmxBlockCc8();
		break;
	case 7:
		smxBlock = GetSmxBlockCc7();
		break;
	case 6:
		smxBlock = GetSmxBlockCc6();
		break;
	case 5:
		smxBlock = GetSmxBlockCc5();
		break;
	case 3:
		smxBlock = GetSmxBlockCc3();
		break;
	default:
		Exit();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc8() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
	case 0:
		smxBlock = 16;	// CC 8.0, occupancy 100% = 16 blocks per SMX
		break;
	case 1:
	default:
		Exit();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc7() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
		case 0:				// CC 7.0 & 7.2, occupancy 100% = 32 blocks per SMX
		case 2:
			smxBlock = 32;
			break;
		case 5:				// CC 7.5, occupancy 100% = 16 blocks per SMX
			smxBlock = 16;
			break;
		default:			
			Exit();
			break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc6() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
		case 0:
		case 1:
		case 2:
			smxBlock = 32; //occupancy 100% = 32 blocks per SMX
			break;
		default:
			Exit();
			break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc5() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
	// TODO: There is something rot in Denmark...
#if (CUDART_VERSION < 11000)
		case 0:
		case 2:
#endif
		case 3:
			smxBlock = 32; //occupancy 100% = 32 blocks per SMX, instead as previous was 16 blocks per SMX which led to only 50%
			break;

		default:
			Exit();
			break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc3() const
{
	auto smxBlock = 0;
	switch(deviceCcMinor)
	{
		//CC 3.0, 3.2, 3.5 & 3.7
		case 0:
		case 2:
		case 3:
		case 5:
		case 7:
			smxBlock = 16; //occupancy 100% = 16 blocks per SMX
			break;
		default:
			Exit();
			break;
	}

	return smxBlock;
}

void Cc::Exit() const
{
	// TODO: Fix those parameters
#if (CUDART_VERSION < 11000)
	fprintf(stderr, "Unsupported Compute Capability (CC) detected (%d.%d). Supported Compute Capabilities are between 3.0 and 7.5.\n", deviceCcMajor, deviceCcMinor);
#elif(CUDART_VERSION >= 11000 && CUDART_VERSION < 12000)
	fprintf(stderr, "Unsupported Compute Capability (CC) detected (%d.%d). Supported Compute Capabilities are between 5.3 and 8.0.\n", deviceCcMajor, deviceCcMinor);
#endif
	exit(1);
}
