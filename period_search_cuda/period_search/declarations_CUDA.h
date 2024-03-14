#pragma once

#define BLOCKX4 4
#define BLOCKX8 8
#define BLOCKX16 16
#define BLOCKX32 32

#define blockIdx() (blockIdx.x + gridDim.x * threadIdx.y)

__device__ void curv(freq_context const *__restrict__ CUDA_LCC,
					 double *__restrict__ cg,
					 int bid);
__device__ int mrqmin_1_end(freq_context *__restrict__ CUDA_LCC,
							int ma, int mfit, int mfit1, int block);
__device__ void mrqmin_2_end(freq_context *__restrict__ CUDA_LCC,
							 int ma, int bid);
__device__ void mrqcof_start(freq_context *__restrict__ CUDA_LCC,
							 double *__restrict__ a,
							 double *__restrict__ alpha,
							 double *__restrict__ beta,
							 int bid);
__device__ void mrqcof_matrix(freq_context *__restrict__ CUDA_LCC,
							  double *__restrict__ a,
							  int Lpoints, int bid);
__device__ void mrqcof_curve1(freq_context *__restrict__ CUDA_LCC,
							  double const *__restrict__ a,
							  int Inrel, int Lpoints, int bid);
__device__ double mrqcof_end(freq_context *__restrict__ CUDA_LCC,
							 double *__restrict__ alpha);
__device__ void mrqcof_curve1_lastI0(freq_context *__restrict__ CUDA_LCC,
									 double *__restrict__ a,
									 double *__restrict__ alpha,
									 double *__restrict__ beta,
									 int bid);
__device__ void mrqcof_curve1_lastI1(freq_context *__restrict__ CUDA_LCC,
									 double *__restrict__ a,
									 double *__restrict__ alpha,
									 double *__restrict__ beta,
									 int bid);
__device__ void MrqcofCurve23I1IA1(freq_context *__restrict__ CUDA_LCC,
								   double *__restrict__ alpha, double *__restrict__ beta, int bid);
__device__ void MrqcofCurve23I0IA0(freq_context *__restrict__ CUDA_LCC,
								   double *__restrict__ alpha, double *__restrict__ beta, int bid);
__device__ void MrqcofCurve23I0IA1(freq_context *__restrict__ CUDA_LCC,
								   double *__restrict__ alpha, double *__restrict__ beta, int bid);
__device__ void MrqcofCurve2I0IA0(freq_context *__restrict__ CUDA_LCC,
								  double *__restrict__ alpha, double *__restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I1IA0(freq_context *__restrict__ CUDA_LCC,
								  double *__restrict__ alpha, double *__restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I0IA1(freq_context *__restrict__ CUDA_LCC,
								  double *__restrict__ alpha, double *__restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I1IA1(freq_context *__restrict__ CUDA_LCC,
								  double *__restrict__ alpha, double *__restrict__ beta, int lpoints, int bid);
__device__ int gauss_errc(freq_context *__restrict__ CUDA_LCC,
						  int ma);
__device__ double conv(freq_context *__restrict__ CUDA_LCC,
					   int nc, double *__restrict__ dyda, int bid);
__device__ double bright(freq_context *__restrict__ CUDA_LCC,
							 double *__restrict__ cg,
							 int jp, int Lpoints1, int Inrel);
__device__ void matrix_neo(freq_context *__restrict__ CUDA_LCC,
						   double const *__restrict__ cg,
						   int lnp1, int Lpoints, int bid);
__device__ void blmatrix(double bet, double lam, int tid);
