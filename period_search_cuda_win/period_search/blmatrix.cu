/* beta, lambda rotation matrix and its derivatives 
   
   8.11.2006
*/

#include <math.h>
#include "globals_CUDA.h"

__device__ void blmatrix(freq_context *CUDA_LCC,double bet, double lam)
{
   double cb, sb, cl, sl;

   cb = cos(bet);
   sb = sin(bet);
   cl = cos(lam);
   sl = sin(lam);
   (*CUDA_LCC).Blmat[1][1] = cb * cl;
   (*CUDA_LCC).Blmat[1][2] = cb * sl;
   (*CUDA_LCC).Blmat[1][3] = -sb;
   (*CUDA_LCC).Blmat[2][1] = -sl;
   (*CUDA_LCC).Blmat[2][2] = cl;
   (*CUDA_LCC).Blmat[2][3] = 0;
   (*CUDA_LCC).Blmat[3][1] = sb * cl;
   (*CUDA_LCC).Blmat[3][2] = sb * sl;
   (*CUDA_LCC).Blmat[3][3] = cb;
   /* Ders. of Blmat w.r.t. bet */
   (*CUDA_LCC).Dblm[1][1][1] = -sb * cl;
   (*CUDA_LCC).Dblm[1][1][2] = -sb * sl;
   (*CUDA_LCC).Dblm[1][1][3] = -cb;
   (*CUDA_LCC).Dblm[1][2][1] = 0;
   (*CUDA_LCC).Dblm[1][2][2] = 0;
   (*CUDA_LCC).Dblm[1][2][3] = 0;
   (*CUDA_LCC).Dblm[1][3][1] = cb * cl;
   (*CUDA_LCC).Dblm[1][3][2] = cb * sl;
   (*CUDA_LCC).Dblm[1][3][3] = -sb;
   /* Ders. w.r.t. lam */
   (*CUDA_LCC).Dblm[2][1][1] = -cb * sl;
   (*CUDA_LCC).Dblm[2][1][2] = cb * cl;
   (*CUDA_LCC).Dblm[2][1][3] = 0;
   (*CUDA_LCC).Dblm[2][2][1] = -cl;
   (*CUDA_LCC).Dblm[2][2][2] = -sl;
   (*CUDA_LCC).Dblm[2][2][3] = 0;
   (*CUDA_LCC).Dblm[2][3][1] = -sb * sl;
   (*CUDA_LCC).Dblm[2][3][2] = sb * cl;
   (*CUDA_LCC).Dblm[2][3][3] = 0;
}
