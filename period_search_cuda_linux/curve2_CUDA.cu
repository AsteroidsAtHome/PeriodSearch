#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"

__device__ void mrqcof_curve2(freq_context *CUDA_LCC, double a[],  
	      double *alpha, double beta[], int Inrel,int Lpoints)
{
   int l,jp,j,k,m, lnp1,lnp2,Lpoints1=Lpoints+1;
   double dy,sig2i,wt,ymod, coef1,coef,  wght,ltrial_chisq;
   int2 xx;

   
//precalc thread boundaries
    int tmph,tmpl;
	tmph=Lpoints/CUDA_BLOCK_DIM;
	if(Lpoints%CUDA_BLOCK_DIM) tmph++;
	tmpl=threadIdx.x*tmph;
	lnp1=(*CUDA_LCC).np1+tmpl;
	tmph=tmpl+tmph;
	if (tmph>Lpoints) tmph=Lpoints;
	tmpl++;

    int matmph,matmpl;
	matmph=CUDA_ma/CUDA_BLOCK_DIM;
	if(CUDA_ma%CUDA_BLOCK_DIM) matmph++;
	matmpl=threadIdx.x*matmph;
	matmph=matmpl+matmph;
	if (matmph>CUDA_ma) matmph=CUDA_ma;
	matmpl++;

    int latmph,latmpl;
	latmph=CUDA_lastone/CUDA_BLOCK_DIM;
	if(CUDA_lastone%CUDA_BLOCK_DIM) latmph++;
	latmpl=threadIdx.x*latmph;
	latmph=latmpl+latmph;
	if (latmph>CUDA_lastone) latmph=CUDA_lastone;
	latmpl++;

/*   if ((*CUDA_LCC).Lastcall != 1) always ==0
   {*/
    if (Inrel /*==1*/) 
	{
      for (jp = tmpl; jp <= tmph; jp++)
      {
        lnp1++;
			int ixx=jp+1*Lpoints1;
				/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
			(*CUDA_LCC).dytemp[ixx]=0;

			xx=tex1Dfetch(texsig,lnp1);
			coef = __hiloint2double(xx.y,xx.x) * Lpoints / (*CUDA_LCC).ave;

			double yytmp=(*CUDA_LCC).ytemp[jp];
			coef1= yytmp / (*CUDA_LCC).ave;
			(*CUDA_LCC).ytemp[jp] = coef * yytmp;

			ixx+=Lpoints1;
			for (l = 2; l <= CUDA_ma; l++,ixx+=Lpoints1)
				(*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);
	  }
	}
	__syncthreads();

   if (threadIdx.x==0) 
   {
   (*CUDA_LCC).np1+=Lpoints;
   }
   //
   lnp2=(*CUDA_LCC).np2;
   ltrial_chisq=(*CUDA_LCC).trial_chisq;

  
   if (CUDA_ia[1]) //not relative
	  { 
		  for (jp = 1; jp <= Lpoints; jp++)
		  {
			 ymod = (*CUDA_LCC).ytemp[jp];

			 int ixx=jp+matmpl*Lpoints1;
			 for (l = matmpl; l <= matmph; l++,ixx+=Lpoints1)
					(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
			 __syncthreads();

			 lnp2++;
			xx=tex1Dfetch(texsig,lnp2);
			sig2i = 1 / (__hiloint2double(xx.y,xx.x) * __hiloint2double(xx.y,xx.x));

			xx=tex1Dfetch(texWeight,lnp2);
			wght =  __hiloint2double(xx.y,xx.x); 

			xx=tex1Dfetch(texbrightness,lnp2);
			dy = __hiloint2double(xx.y,xx.x) - ymod;

			 j = 0;
			 //
			 double sig2iwght=sig2i * wght;
			 //
			 for (l = 1; l <= CUDA_lastone; l++)
			 {
				   j++;
				   wt = (*CUDA_LCC).dyda[l] * sig2iwght;
//				   k = 0;
//precalc thread boundaries
					tmph=l/CUDA_BLOCK_DIM;
					if(l%CUDA_BLOCK_DIM) tmph++;
					tmpl=threadIdx.x*tmph;
					tmph=tmpl+tmph;
					if (tmph>l) tmph=l;
					tmpl++;
				   for (m = tmpl; m <= tmph; m++)
				   {
	//				  k++;
					  alpha[j*(CUDA_mfit1)+m] = alpha[j*(CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
				   } /* m */
				   __syncthreads();
				   if (threadIdx.x==0)
				   {
					   beta[j] = beta[j] + dy * wt;
				   }
				   __syncthreads();
			 } /* l */ 
			 for (; l <= CUDA_lastma; l++)
			 {
				if(CUDA_ia[l]) 
				{
				   j++;
				   wt = (*CUDA_LCC).dyda[l] * sig2iwght;
//				   k = 0;

				   for (m = latmpl; m <= latmph; m++)
				   {
//					  k++;
					  alpha[j*(CUDA_mfit1)+m] = alpha[j*(CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
				   } /* m */
				   __syncthreads();
				   if (threadIdx.x==0)
				   {
					   k=CUDA_lastone;
					   m=CUDA_lastone+1;
					   for (; m <= l; m++)
					   {
						  if(CUDA_ia[m])
						  {
							 k++;
							 alpha[j*(CUDA_mfit1)+k] = alpha[j*(CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
						  }
					   } /* m */
					   beta[j] = beta[j] + dy * wt;
				   }
				   __syncthreads();
				  }
			 } /* l */ 
			 ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
			} /* jp */
	  }
	  else //relative ia[1]==0
	  {
		  for (jp = 1; jp <= Lpoints; jp++)
		  {
			 ymod = (*CUDA_LCC).ytemp[jp];

			 int ixx=jp+matmpl*Lpoints1;
			 for (l = matmpl; l <= matmph; l++,ixx+=Lpoints1)
					(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
			 __syncthreads();

			 lnp2++;

			xx=tex1Dfetch(texsig,lnp2);
			sig2i = 1 / (__hiloint2double(xx.y,xx.x) * __hiloint2double(xx.y,xx.x));

			xx=tex1Dfetch(texWeight,lnp2);
			wght =  __hiloint2double(xx.y,xx.x); 

			xx=tex1Dfetch(texbrightness,lnp2);
			dy = __hiloint2double(xx.y,xx.x) - ymod;

			 j = 0;
			 //
			 double sig2iwght=sig2i * wght;
			 //l==1
			 //
			 for (l = 2; l <= CUDA_lastone; l++)
			 {
				   j++;
				   wt = (*CUDA_LCC).dyda[l] * sig2iwght;
//				   k = 0;
//precalc thread boundaries
					tmph=l/CUDA_BLOCK_DIM;
					if(l%CUDA_BLOCK_DIM) tmph++;
					tmpl=threadIdx.x*tmph;
					tmph=tmpl+tmph;
					if (tmph>l) tmph=l;
					tmpl++;
				   //m==1
					if (tmpl==1) tmpl++;
				   //
				   for (m = tmpl; m <= tmph; m++)
				   {
//					  k++;
					  alpha[j*(CUDA_mfit1)+m-1] = alpha[j*(CUDA_mfit1)+m-1] + wt * (*CUDA_LCC).dyda[m];
				   } /* m */
				   __syncthreads();
				   if (threadIdx.x==0)
				   {
						beta[j] = beta[j] + dy * wt;
				   }
				  __syncthreads();
			 } /* l */ 
			 for (; l <= CUDA_lastma; l++)
			 {
				if(CUDA_ia[l]) 
				{
				   j++;
				   wt = (*CUDA_LCC).dyda[l] * sig2iwght;
//				   k = 0;

				   tmpl=latmpl;
					//m==1
					if (tmpl==1) tmpl++;
				   //
				   for (m = tmpl; m <= latmph; m++)
				   {
					  //k++;
					  alpha[j*(CUDA_mfit1)+m-1] = alpha[j*(CUDA_mfit1)+m-1] + wt * (*CUDA_LCC).dyda[m];
				   } /* m */
				   __syncthreads();
				   if (threadIdx.x==0)
				   {
					   k=CUDA_lastone-1;
					   m=CUDA_lastone+1;
					   for (; m <= l; m++)
					   {
						  if(CUDA_ia[m])
						  {
							 k++;
							 alpha[j*(CUDA_mfit1)+k] = alpha[j*(CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
						  }
					   } /* m */
					   beta[j] = beta[j] + dy * wt;
				  }
				  __syncthreads();
				}
			 } /* l */ 
			 ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
			} /* jp */
	  }
/*     } always ==0 /* Lastcall != 1 */
         
   /*  if (((*CUDA_LCC).Lastcall == 1) && (CUDA_Inrel[i] == 1)) always ==0
        (*CUDA_LCC).Sclnw[i] = (*CUDA_LCC).Scale * CUDA_Lpoints[i] * CUDA_sig[np]/ave;*/
	if(threadIdx.x==0)
	{
		(*CUDA_LCC).np2=lnp2;
		(*CUDA_LCC).trial_chisq=ltrial_chisq;
	}
}


__global__ void CUDACalculateIter1_mrqcof1_curve2(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
    freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve2(CUDA_LCC,(*CUDA_LCC).cg,(*CUDA_LCC).alpha,(*CUDA_LCC).beta,Inrel,Lpoints);
}

__global__ void CUDACalculateIter1_mrqcof2_curve2(int Inrel,int Lpoints)
{
	int thidx=blockIdx.x;
	freq_context *CUDA_LCC=&CUDA_CC[thidx];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve2(CUDA_LCC,(*CUDA_LCC).atry,(*CUDA_LCC).covar,(*CUDA_LCC).da,Inrel,Lpoints);
}
