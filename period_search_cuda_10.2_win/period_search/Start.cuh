#pragma once

__global__ void CUDACalculatePrepare(int n_start, int n_max, double freq_start, double freq_step);

__global__ void CUDACalculatePreparePole(int m);

__global__ void CUDACalculateIter1_Begin(void);

__global__ void CUDACalculateIter1_mrqmin1_end(void);

__global__ void CUDACalculateIter1_mrqmin2_end(void);

__global__ void CUDACalculateIter1_mrqcof1_start(void);

__global__ void CUDACalculateIter1_mrqcof1_matrix(int Lpoints);

__global__ void CUDACalculateIter1_mrqcof1_curve1(int Inrel, int Lpoints);

__global__ void CUDACalculateIter1_mrqcof1_curve1_last(int Inrel, int Lpoints);

__global__ void CUDACalculateIter1_mrqcof1_end(void);

__global__ void CUDACalculateIter1_mrqcof2_start(void);

__global__ void CUDACalculateIter1_mrqcof2_matrix(int Lpoints);

__global__ void CUDACalculateIter1_mrqcof2_curve1(int Inrel, int Lpoints);

__global__ void CUDACalculateIter1_mrqcof2_curve1_last(int Inrel, int Lpoints);

__global__ void CUDACalculateIter1_mrqcof2_end(void);

__global__ void CUDACalculateIter2(void);

__global__ void CUDACalculateFinishPole(void);

__global__ void CUDACalculateFinish(void);