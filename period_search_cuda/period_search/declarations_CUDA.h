#pragma once
__device__ void curv(freq_context *CUDA_LCC,double cg[],int brtmpl,int brtmph);
extern __device__ int mrqmin_1_end(freq_context *CUDA_LCC, int ma, int mfit, int mfit1, int block);
__device__ void mrqmin_2_end(freq_context *CUDA_LCC, int ia[], int ma);
__device__ void mrqcof_start(freq_context *CUDA_LCC, double a[],
                             double *alpha, double beta[]);
__device__ void mrqcof_matrix(freq_context *CUDA_LCC, double a[], int Lpoints);
__device__ void mrqcof_curve1(freq_context *CUDA_LCC, double a[],
                              double *alpha, double beta[], int Inrel,int Lpoints);
__device__ void mrqcof_curve1_last(freq_context *CUDA_LCC, double a[], double *alpha, double beta[], int Inrel,int Lpoints);
__device__ void MrqcofCurve2(freq_context *CUDA_LCC, double *alpha, double beta[], int inrel,int lpoints);
__device__ double mrqcof_end(freq_context *CUDA_LCC,  double *alpha);

__device__ double mrqcof(freq_context *CUDA_LCC, double a[], int ia[], int ma,
                         double alpha[/*MAX_N_PAR+1*/][MAX_N_PAR+1], double beta[], int mfit, int lastone, int lastma);
//__device__ int gauss_errc(freq_context *CUDA_LCC,int n, double b[]);
extern __device__ int gauss_errc_shared(freq_context *CUDA_LCC, int ma);
__device__ void blmatrix(freq_context *CUDA_LCC,double bet, double lam);
__device__ void bright_curve1_warp(freq_context *CUDA_LCC, double const *a, int Inrel, int Lpoints);
__global__ void CudaCalculateIter1Mrqcof2Curve2(int inrel,int lpoints);
__global__ void CudaCalculateIter1Mrqcof1Curve2(int inrel,int lpoints);
