#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include "ComputeCapability.h"

Cc::Cc(const cudaDeviceProp& deviceProp)
{
	this->cudaVersion = CUDART_VERSION;
	deviceCcMajor = deviceProp.major;
	deviceCcMinor = deviceProp.minor;
}

#if defined (_MSC_VER) & (_MSC_VER < 1900) // Visual Studio 2012 or previous
	Cc::~Cc(){}
#else
#if defined __GNUC__
	Cc::~Cc() {}
#endif

#endif


int Cc::GetSmxBlock() const
{
	auto result = 0;
	if (cudaVersion >= 12000 && cudaVersion < 13000)
	{
		result = GetSmxBlockCuda12();
	}
	else if (cudaVersion >= 11000 && cudaVersion < 12000)
	{
		result = GetSmxBlockCuda11();
	}
	else if (cudaVersion >= 10000 && cudaVersion < 11000)
	{
		result = GetSmxBlockCuda10();
	}
	else if (cudaVersion >= 6000 && cudaVersion < 10000)
	{
		result = GetSmxBlockCuda6();
	}
	else
	{
		result = UseDefault();
	}

	return result;
}


int Cc::GetSmxBlockCuda12() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 12:
		smxBlock = GetSmxBlockCc12(); // Blackwell
		break;
	case 10:
		smxBlock = 16; // Fall back to safe value
		break;
	case 9:
		smxBlock = GetSmxBlockCc9(); // Hopper
		break;
	case 8:
		smxBlock = GetSmxBlockCc8(); // Ampere micro architecture CC 8.0, 8.6; Ada Lovelace - CC 8.9
		break;
	case 7:
		smxBlock = GetSmxBlockCc7(); // 7.0, 7.2: Volta; 7.5: Turing 
		break;
	case 6:
		smxBlock = GetSmxBlockCc6(); // Pascal
		break;
	case 5:
		smxBlock = GetSmxBlockCc5(); // Maxwell
		break;
	default:
		smxBlock = UseDefault();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCuda11() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 10:
		smxBlock = 16; // Fall back to safe value
		break;
	case 9:
		smxBlock = GetSmxBlockCc9(); // Hopper
		break;
	case 8:
		smxBlock = GetSmxBlockCc8(); // Ampere micro architecture CC 8.0, 8.6; Ada Lovelace - CC 8.9
		break;
	case 7:
		smxBlock = GetSmxBlockCc7(); // 7.0, 7.2: Volta; 7.5: Turing 
		break;
	case 6:
		smxBlock = GetSmxBlockCc6(); // Pascal
		break;
	case 5:
		smxBlock = GetSmxBlockCc5(); // Maxwell
		break;
	default:
		smxBlock = UseDefault();
		break;
	}

	return smxBlock;
}



int Cc::GetSmxBlockCuda10() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 10:
	case 9:
		smxBlock = 16; // Fall back to safe value
		break;
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
		smxBlock = GetSmxBlockCc3(); // Kepler
		break;
	default:
		smxBlock = UseDefault();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCuda6() const
{
	auto smxBlock = 0;
	switch (deviceCcMajor)
	{
	case 10:
	case 9:
	case 8:
	case 7:
	case 6:
	case 5:
		smxBlock = 16; // Fall back to safe value 
		break;
	case 3:
		smxBlock = GetSmxBlockCc3(); // Kepler
		break;
	case 2:
		smxBlock = GetSmxBlockCc2(); // Fermi
		break;
	case 1:
		smxBlock = GetSmxBlockCc1(); // Tesla
		break;
	default:
		smxBlock = UseDefault();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc12() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
	case 0:
		smxBlock = 32;	// Blackwell
		break;
	case 1:
		smxBlock = 32;	// Blackwell (GB10)
		break;
	default:
		smxBlock = UseDefault();
		break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc9() const
{
	auto smxBlock = 0;
	switch (deviceCcMinor)
	{
	case 0:
		smxBlock = 32;	// Hopper
		break;
	default:
		smxBlock = UseDefault();
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
		smxBlock = 32;	// Tesla A100 | occupancy 100% = 32 blocks per SMX
		break;
	case 5:
		smxBlock = 16;	// ZLuda
		break;
	case 6:
	case 7:
		smxBlock = 16;	// GeForce RTX 3080 etc.; Quadro A6000 | occupancy 100% = 16 blocks per SMX
		break;
	case 8:
		smxBlock = 16;	// ZLuda
		break;
	case 9:
		smxBlock = 24;	// GeForce RTX 4090, RTX 4080 16GB; RTX 6000 Ada | occupancy 100% = 24 blocks per SMX
		break;
	default:
		smxBlock = UseDefault();
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
			smxBlock = UseDefault();
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
			smxBlock = UseDefault();
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
//#if (CUDART_VERSION < 11000)
		case 0:
		case 2:
//#endif
		case 3:
			smxBlock = 32; //occupancy 100% = 32 blocks per SMX, instead as previous was 16 blocks per SMX which led to only 50%
			break;
		default:
			smxBlock = UseDefault();
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
			smxBlock = UseDefault();
			break;			
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc2() const
{
	auto smxBlock = 0;
	switch(deviceCcMinor)
	{
		//CC 2.0, 2.1
		case 0:
		case 1:
			smxBlock = 8; //occupancy 100% = 8 blocks per SMX
			break;
		default:
			smxBlock = UseDefault();
			break;
	}

	return smxBlock;
}

int Cc::GetSmxBlockCc1() const
{
	auto smxBlock = 0;
	switch(deviceCcMinor)
	{
		//CC 1.0, 1.1, 1.2, 1.3
		case 0:
		case 1:
		case 2:
		case 3:
			smxBlock = 8; //occupancy 100% = 8 blocks per SMX
			break;
		default:
			smxBlock = UseDefault();
			break;
	}

	return smxBlock;
}

int Cc::UseDefault() const
{
	fprintf(stderr, "Warning: Unsupported Compute Capability (CC) detected (%d.%d) with CUDA runtime %d. Using default of %d resident blocks per SMX.\n",
		deviceCcMajor, deviceCcMinor, cudaVersion, DefaultSmxBlock);
	return DefaultSmxBlock;
}
