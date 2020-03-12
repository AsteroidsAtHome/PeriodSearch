/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>


/* comment the following line if no YORP */
/*#define YORP*/

__device__ void mrqcof_start(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[])
{
   int j,k;
//
    int brtmph,brtmpl;
	brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
	if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
	brtmpl++;

   /* N.B. curv and blmatrix called outside bright
      because output same for all points */
   curv(CUDA_LCC,a,brtmpl,brtmph);

   if (threadIdx.x==0)
   {
//   #ifdef YORP
//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
  // #else
      blmatrix(CUDA_LCC,a[CUDA_ma-4-CUDA_Nphpar],a[CUDA_ma-3-CUDA_Nphpar]);
//   #endif
	   (*CUDA_LCC).trial_chisq = 0;
	   (*CUDA_LCC).np = 0;
	   (*CUDA_LCC).np1 = 0;
	   (*CUDA_LCC).np2 = 0;
	   (*CUDA_LCC).ave = 0;
   }

    brtmph=CUDA_mfit/CUDA_BLOCK_DIM;
	if(CUDA_mfit%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_mfit) brtmph=CUDA_mfit;
	brtmpl++;

   for(j = brtmpl; j <= brtmph; j++)
   {
      for (k = 1; k <= j; k++)
         alpha[j*(CUDA_mfit1)+k]=0;
      beta[j]=0;
   }

   __syncthreads(); //pro jistotu
}

__device__ double mrqcof_end(freq_context *CUDA_LCC,double *alpha)
{
   int j,k;

   for (j = 2; j <= CUDA_mfit; j++)
      for (k = 1; k <= j-1; k++)
         alpha[k*(CUDA_mfit1)+j] = alpha[j*(CUDA_mfit1)+k];

   return (*CUDA_LCC).trial_chisq;
}

__device__ void mrqcof_matrix(freq_context *CUDA_LCC, double a[], int Lpoints)
{
   matrix_neo(CUDA_LCC, a,(*CUDA_LCC).np, Lpoints);
}

__device__ void mrqcof_curve1(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,k,jp, lnp,Lpoints1=Lpoints+1;
   double lave;
   __shared__ double tmave[CUDA_BLOCK_DIM];

   lnp=(*CUDA_LCC).np;
   lave=(*CUDA_LCC).ave;
//precalc thread boundaries
    int brtmph,brtmpl;
	brtmph=Lpoints/CUDA_BLOCK_DIM;
	if(Lpoints%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>Lpoints) brtmph=Lpoints;
	brtmpl++;
//

   for (jp = brtmpl; jp <= brtmph; jp++)
    bright(CUDA_LCC,a,jp,Lpoints1,Inrel);

   __syncthreads();

  if (Inrel == 1) {
    int tmph,tmpl;
	tmph=CUDA_ma/CUDA_BLOCK_DIM;
	if(CUDA_ma%CUDA_BLOCK_DIM) tmph++;
	tmpl=threadIdx.x*tmph;
	tmph=tmpl+tmph;
	if (tmph>CUDA_ma) tmph=CUDA_ma;
	tmpl++;
	if (tmpl==1) tmpl++;

	  int ixx;
	  ixx=tmpl*Lpoints1;
	  for (l=tmpl; l <= tmph; l++)
		{
	  //jp==1
			ixx++;
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];
      //jp>=2
			ixx++;
		   for (jp = 2; jp <= Lpoints; jp++,ixx++)
		   {
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
		   }

	  }
		tmave[threadIdx.x] = 0;
	   for (jp = brtmpl; jp <= brtmph; jp++) tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
	   __syncthreads();
//parallel reduction
	   k=CUDA_BLOCK_DIM>>1;
	   while (k>1)
	   {
		   if (threadIdx.x<k) tmave[threadIdx.x]+=tmave[threadIdx.x+k];
		   k=k>>1;
		   __syncthreads();
	   }
	   if (threadIdx.x==0) lave=tmave[0]+tmave[1];
//parallel reduction end

  }
	  if (threadIdx.x==0)
	  {
	   (*CUDA_LCC).np=lnp+Lpoints;
	   (*CUDA_LCC).ave=lave;
	  }
}

__device__ void mrqcof_curve1_last(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,jp, lnp;
   double ymod, lave;

   lnp=(*CUDA_LCC).np;
   //
   if (threadIdx.x==0)
   {
	   if (Inrel == 1) /* is the LC relative? */
	   {
		  lave = 0;
		  for (l = 1; l <= CUDA_ma; l++)
		  (*CUDA_LCC).dave[l]=0;
	   }
	   else
		  lave=(*CUDA_LCC).ave;
   }
//precalc thread boundaries
    int tmph,tmpl;
	tmph=CUDA_ma/CUDA_BLOCK_DIM;
	if(CUDA_ma%CUDA_BLOCK_DIM) tmph++;
	tmpl=threadIdx.x*tmph;
	tmph=tmpl+tmph;
	if (tmph>CUDA_ma) tmph=CUDA_ma;
	tmpl++;
//
    int brtmph,brtmpl;
	brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
	if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
	brtmpl++;

	__syncthreads();


      for (jp = 1; jp <= Lpoints; jp++)
      {
         lnp++;

         ymod = conv(CUDA_LCC,jp-1,tmpl,tmph,brtmpl,brtmph);

		 if (threadIdx.x==0)
		 {
			 (*CUDA_LCC).ytemp[jp] = ymod;

			 if (Inrel == 1)
				lave = lave + ymod;
		 }
		for (l=tmpl; l <= tmph; l++)
		{
			(*CUDA_LCC).dytemp[jp+l*(Lpoints+1)] = (*CUDA_LCC).dyda[l];
			if (Inrel == 1)
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
		}
		/* save lightcurves */
		 __syncthreads();

/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
			 (*CUDA_LCC).Yout[np] = ymod;*/
      } /* jp, lpoints */
	 if (threadIdx.x==0)
	 {
		  (*CUDA_LCC).np=lnp;
		  (*CUDA_LCC).ave=lave;
	 }
}

