#pragma once

__global__ void CudaCalculatePrepare(int n_start, int n_max, double freq_start, double freq_step, const int nContexts);

__global__ void CudaCalculatePreparePole(const int nContexts);

__global__ void CudaCalculateIter1Begin(const int nContexts);

__global__ void CudaCalculateIter1Mrqmin1End(void);

__global__ void CudaCalculateIter1Mrqmin2End(void);

__global__ void CudaCalculateIter1Mrqcof1Start(void);

__global__ void CudaCalculateIter1Mrqcof1Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1Last(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof1End(void);

__global__ void CudaCalculateIter1Mrqcof2Start(void);

__global__ void CudaCalculateIter1Mrqcof2Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1Last(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof2End(void);

__global__ void CudaCalculateIter2(void);

__global__ void CudaCalculateFinishPole(const int nContexts);

__global__ void CudaCalculateFinish(const int nContexts);