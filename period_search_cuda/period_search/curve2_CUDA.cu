//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void MrqcofCurve2(freq_context* CUDA_LCC, double* alpha, double beta[], int inrel, int lpoints)
{
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  //int2 xx;


  //precalc thread boundaries
  int tmph, tmpl;
  tmph = lpoints / CUDA_BLOCK_DIM;
  if (lpoints % CUDA_BLOCK_DIM) tmph++;
  tmpl = threadIdx.x * tmph;
  lnp1 = (*CUDA_LCC).np1 + tmpl;
  tmph = tmpl + tmph;
  if (tmph > lpoints) tmph = lpoints;
  tmpl++;

  int matmph, matmpl;
  matmph = CUDA_ma / CUDA_BLOCK_DIM;
  if (CUDA_ma % CUDA_BLOCK_DIM) matmph++;
  matmpl = threadIdx.x * matmph;
  matmph = matmpl + matmph;
  if (matmph > CUDA_ma) matmph = CUDA_ma;
  matmpl++;

  int latmph, latmpl;
  latmph = CUDA_lastone / CUDA_BLOCK_DIM;
  if (CUDA_lastone % CUDA_BLOCK_DIM) latmph++;
  latmpl = threadIdx.x * latmph;
  latmph = latmpl + latmph;
  if (latmph > CUDA_lastone) latmph = CUDA_lastone;
  latmpl++;

  /*   if ((*CUDA_LCC).Lastcall != 1) always ==0
       {*/
  if (inrel /*==1*/)
    {
      for (jp = tmpl; jp <= tmph; jp++)
	{
	  lnp1++;
	  int ixx = jp + 1 * Lpoints1;
	  /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
	  (*CUDA_LCC).dytemp[ixx] = 0;

	  //xx = tex1Dfetch(texsig, lnp1);
	  //coef = __hiloint2double(xx.y, xx.x) * lpoints / (*CUDA_LCC).ave;
	  coef = CUDA_sig[lnp1] * lpoints / (*CUDA_LCC).ave;

	  double yytmp = (*CUDA_LCC).ytemp[jp];
	  coef1 = yytmp / (*CUDA_LCC).ave;
	  (*CUDA_LCC).ytemp[jp] = coef * yytmp;

	  ixx += Lpoints1;
	  for (l = 2; l <= CUDA_ma; l++, ixx += Lpoints1)
	    (*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);
	}
    }
  __syncthreads();

  if (threadIdx.x == 0)
    {
      (*CUDA_LCC).np1 += lpoints;
    }

  lnp2 = (*CUDA_LCC).np2;
  ltrial_chisq = (*CUDA_LCC).trial_chisq;

  if (CUDA_ia[1]) //not relative
    {
      for (jp = 1; jp <= lpoints; jp++)
	{
	  ymod = (*CUDA_LCC).ytemp[jp];

	  int ixx = jp + matmpl * Lpoints1;
	  for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
	    (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
	  __syncthreads();

	  lnp2++;
			
	  //xx = tex1Dfetch(texsig, lnp2);
	  //sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));
	  sig2i = 1 / (CUDA_sig[lnp2] * CUDA_sig[lnp2]);

	  //xx = tex1Dfetch(texWeight, lnp2);
	  //wght = __hiloint2double(xx.y, xx.x);
	  wght = CUDA_Weight[lnp2];

	  //xx = tex1Dfetch(texbrightness, lnp2);
	  //dy = __hiloint2double(xx.y, xx.x) - ymod;
	  dy = CUDA_brightness[lnp2] - ymod;

	  j = 0;
	  //
	  double sig2iwght = sig2i * wght;
	  //
	  for (l = 1; l <= CUDA_lastone; l++)
	    {
	      j++;
	      wt = (*CUDA_LCC).dyda[l] * sig2iwght;
	      //				   k = 0;
	      //precalc thread boundaries
	      tmph = l / CUDA_BLOCK_DIM;
	      if (l % CUDA_BLOCK_DIM) tmph++;
	      tmpl = threadIdx.x * tmph;
	      tmph = tmpl + tmph;
	      if (tmph > l) tmph = l;
	      tmpl++;
	      for (m = tmpl; m <= tmph; m++)
		{
		  //				  k++;
		  alpha[j * (CUDA_mfit1)+m] = alpha[j * (CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
		} /* m */
	      __syncthreads();
	      if (threadIdx.x == 0)
		{
		  beta[j] = beta[j] + dy * wt;
		}
	      __syncthreads();
	    } /* l */
	  for (; l <= CUDA_lastma; l++)
	    {
	      if (CUDA_ia[l])
		{
		  j++;
		  wt = (*CUDA_LCC).dyda[l] * sig2iwght;
		  //				   k = 0;

		  for (m = latmpl; m <= latmph; m++)
		    {
		      //					  k++;
		      alpha[j * (CUDA_mfit1)+m] = alpha[j * (CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
		    } /* m */
		  __syncthreads();
		  if (threadIdx.x == 0)
		    {
		      k = CUDA_lastone;
		      m = CUDA_lastone + 1;
		      for (; m <= l; m++)
			{
			  if (CUDA_ia[m])
			    {
			      k++;
			      alpha[j * (CUDA_mfit1)+k] = alpha[j * (CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
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
      for (jp = 1; jp <= lpoints; jp++)
	{
	  ymod = (*CUDA_LCC).ytemp[jp];

	  int ixx = jp + matmpl * Lpoints1;
	  for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
	    (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
	  __syncthreads();

	  lnp2++;

	  //xx = tex1Dfetch(texsig, lnp2);
	  //sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));
	  sig2i = 1 / (CUDA_sig[lnp2] * CUDA_sig[lnp2]);

	  //xx = tex1Dfetch(texWeight, lnp2);
	  //wght = __hiloint2double(xx.y, xx.x);
	  wght = CUDA_Weight[lnp2];

	  //xx = tex1Dfetch(texbrightness, lnp2);
	  //dy = __hiloint2double(xx.y, xx.x) - ymod;
	  dy = CUDA_brightness[lnp2] - ymod;

	  j = 0;
	  //
	  double sig2iwght = sig2i * wght;
	  //l==1
	  //
	  for (l = 2; l <= CUDA_lastone; l++)
	    {
	      j++;
	      wt = (*CUDA_LCC).dyda[l] * sig2iwght;
	      //				   k = 0;
	      //precalc thread boundaries
	      tmph = l / CUDA_BLOCK_DIM;
	      if (l % CUDA_BLOCK_DIM) tmph++;
	      tmpl = threadIdx.x * tmph;
	      tmph = tmpl + tmph;
	      if (tmph > l) tmph = l;
	      tmpl++;
	      //m==1
	      if (tmpl == 1) tmpl++;
	      //
	      for (m = tmpl; m <= tmph; m++)
		{
		  //					  k++;
		  alpha[j * (CUDA_mfit1)+m - 1] = alpha[j * (CUDA_mfit1)+m - 1] + wt * (*CUDA_LCC).dyda[m];
		} /* m */
	      __syncthreads();
	      if (threadIdx.x == 0)
		{
		  beta[j] = beta[j] + dy * wt;
		}
	      __syncthreads();
	    } /* l */
	  for (; l <= CUDA_lastma; l++)
	    {
	      if (CUDA_ia[l])
		{
		  j++;
		  wt = (*CUDA_LCC).dyda[l] * sig2iwght;
		  //				   k = 0;

		  tmpl = latmpl;
		  //m==1
		  if (tmpl == 1) tmpl++;
		  //
		  for (m = tmpl; m <= latmph; m++)
		    {
		      //k++;
		      alpha[j * (CUDA_mfit1)+m - 1] = alpha[j * (CUDA_mfit1)+m - 1] + wt * (*CUDA_LCC).dyda[m];
		    } /* m */
		  __syncthreads();
		  if (threadIdx.x == 0)
		    {
		      k = CUDA_lastone - 1;
		      m = CUDA_lastone + 1;
		      for (; m <= l; m++)
			{
			  if (CUDA_ia[m])
			    {
			      k++;
			      alpha[j * (CUDA_mfit1)+k] = alpha[j * (CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
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

  if (threadIdx.x == 0)
    {
      (*CUDA_LCC).np2 = lnp2;
      (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
}


__global__ void CudaCalculateIter1Mrqcof1Curve2(const int inrel, const int lpoints)
{
  const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

  if ((*CUDA_LCC).isInvalid) return;

  if (!(*CUDA_LCC).isNiter) return;

  if (!(*CUDA_LCC).isAlamda) return;

  MrqcofCurve2(CUDA_LCC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2(const int inrel, const int lpoints)
{
  const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

  if ((*CUDA_LCC).isInvalid) return;

  if (!(*CUDA_LCC).isNiter) return;

  MrqcofCurve2(CUDA_LCC, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}
