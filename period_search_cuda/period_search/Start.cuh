#pragma once

__global__ void CudaCalculatePrepare(int n_start, int n_max);

__global__ void CudaCalculatePreparePole(int m, double freq_start, double freq_step, int n);

__global__ void CudaCalculateIter1Begin(int n_max);

__global__ void CudaCalculateIter1Mrqmin1End(void);

__global__ void CudaCalculateIter1Mrqmin2End(void);

__global__ void CudaCalculateIter1Mrqcof1Start(void);

__global__ void CudaCalculateIter1Mrqcof1Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA1(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA1(void);

__global__ void CudaCalculateIter1Mrqcof1CurveM1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1LastI0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve1LastI1(void);

__global__ void CudaCalculateIter1Mrqcof1End(void);

__global__ void CudaCalculateIter1Mrqcof2Start(void);

__global__ void CudaCalculateIter1Mrqcof2Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1I0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1I1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA1(void);

__global__ void CudaCalculateIter1Mrqcof2CurveM1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1LastI1(void);

__global__ void CudaCalculateIter1Mrqcof2Curve1LastI0(void);

__global__ void CudaCalculateIter1Mrqcof2End(void);

__global__ void CudaCalculateIter2(void);

__global__ void CudaCalculateFinishPole(void);

__global__ void CudaCalculateFinish(void);