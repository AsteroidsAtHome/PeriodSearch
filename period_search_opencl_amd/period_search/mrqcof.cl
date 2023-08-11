 //slighly changed code from Numerical Recipes
 //  converted from Mikko's fortran code

 //  8.11.2006


//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"


/* comment the following line if no YORP */
/*#define YORP*/

void mrqcof_start(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__global double* alpha,
	__global double* beta)
{
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	int x = threadIdx.x;

	int brtmph, brtmpl;
	// brtmph = 288 / 128 = 2 (2.25)
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM)
	{
		brtmph++; // brtmph = 3
	}

	brtmpl = threadIdx.x * brtmph;	// 0 * 3 = 0, 1 * 3 = 3, 6,  9, 12, 15, 18... 381(127 * 3)
	brtmph = brtmpl + brtmph;		//		   3,         6, 9, 12, 15, 18, 21... 384(381 + 3)
	if (brtmph > (*CUDA_CC).Numfac) //  97 * 3 = 201 > 288
	{
		brtmph = (*CUDA_CC).Numfac; // 3, 6, ... max 288
	}

	brtmpl++; // 1..382
	//if(blockIdx.x == 0)
	//	printf("Idx: %d | Numfac: %d | brtmpl: %d | brtmph: %d\n", threadIdx.x, (*CUDA_CC).Numfac, brtmpl, brtmph);

		/*  ---   CURV  ---  */
	curv(CUDA_LCC, CUDA_CC, cg, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		//   #ifdef YORP
		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
		  // #else

		//if (blockIdx.x == 0)
		//	printf("[mrqcof_start] a[%3d]: %10.7f, a[%3d]: %10.7f\n",
		//		(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar],
		//		(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);

		  /*  ---  BLMATRIX ---  */
		blmatrix(CUDA_LCC, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar], cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);
		//   #endif
		(*CUDA_LCC).trial_chisq = 0.0;
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = 0;
	}

	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	__private int idx, k, j;

	for (j = brtmpl; j <= brtmph; j++)
	{
		for (k = 1; k <= j; k++)
		{
			idx = j * (*CUDA_CC).Mfit1 + k;
			alpha[idx] = 0;
			//if (blockIdx.x == 0 && j < 3)
			//	printf("[%3d] j: %d, k: %d, Mfit1: %2d, alpha[%3d]: %.7f\n", threadIdx.x, j, k, (*CUDA_CC).Mfit1, idx, alpha[idx]);
		}
		beta[j] = 0;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads(); //pro jistotu

	//int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%d][%3d] cg[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, q, (*CUDA_LCC).cg[q]);


}

void mrqcof_matrix(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int Lpoints,
	int num)
{
	matrix_neo(CUDA_LCC, CUDA_CC, cg, (*CUDA_LCC).np, Lpoints, num);
}

void mrqcof_curve1(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__local double* tmave,
	int Inrel,
	int Lpoints,
	int num)
{
	//__local double tmave[BLOCK_DIM];  // __shared__
	__private int Lpoints1 = Lpoints + 1;
	__private int k, lnp, jp;
	__private double lave;

	lnp = (*CUDA_LCC).np;
	lave = (*CUDA_LCC).ave;

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//precalc thread boundaries
	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	for (jp = brtmpl; jp <= brtmph; jp++)
	{
		//if (num == 1 && blockIdx.x == 2)
		//	printf("[%d][%d] jp: %d\n", blockIdx.x, threadIdx.x, jp);
			//if (blockIdx.x == 0)
			//	printf("[%3d][%3d] brtmpl: %3d, brtmph: %3d\n", blockIdx.x, threadIdx.x, brtmpl, brtmph);

		//if (blockIdx.x == 0 && (jp == 127 || jp == 130))
		//	printf("jp[%d] dytemp[8636]: %10.7f\n", jp, dytemp[8636]);

			/*  ---  BRIGHT  ---  */
		bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel);
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	if (Inrel == 1)
	{
		int tmph, tmpl;
		tmph = (*CUDA_CC).ma / BLOCK_DIM;
		if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
		tmpl = threadIdx.x * tmph;
		tmph = tmpl + tmph;
		if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
		tmpl++;
		if (tmpl == 1) tmpl++;

		int ixx;
		ixx = tmpl * Lpoints1;

		for (int l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx++;
			//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];

			// dytemp[315] -> OK
			//if (threadIdx.x == 1)
			//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, (*CUDA_LCC).dytemp[ixx], l, (*CUDA_LCC).dave[l]);

			//jp>=2
			ixx++;
			for (int jp = 2; jp <= Lpoints; jp++, ixx++)
			{
				//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];

				//if (threadIdx.x == 1)
				//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, (*CUDA_LCC).dytemp[ixx], l, (*CUDA_LCC).dave[l]);
			}
		}

		tmave[threadIdx.x] = 0;
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];

			//if (threadIdx.x == 0) //blockIdx.x == 1 &&
			//	printf("[%d][%3d] ytemp[%d]: %10.7f, tmave[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).ytemp[jp], threadIdx.x, tmave[threadIdx.x]);
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		//parallel reduction
		k = BLOCK_DIM >> 1;
		while (k > 1)
		{
			if (threadIdx.x < k) tmave[threadIdx.x] += tmave[threadIdx.x + k];
			k = k >> 1;
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			lave = tmave[0] + tmave[1];

			//if (blockIdx.x == 2 && num == 1)
			 //   printf("[%d][%d]  \tlave: % .6f\n", blockIdx.x, threadIdx.x, lave);
		}
		//parallel reduction end
	}
	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp + Lpoints;
		(*CUDA_LCC).ave = lave;
	}
}

void mrqcof_curve1_last(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* a,
	__global double* alpha,
	__global double* beta,
	__local double* res,
	int Inrel,
	int Lpoints)
{
	int l, jp, lnp;
	double ymod, lave;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	lnp = (*CUDA_LCC).np;
	//
	if (threadIdx.x == 0)
	{
		if (Inrel == 1) /* is the LC relative? */
		{
			lave = 0;
			for (l = 1; l <= (*CUDA_CC).ma; l++)
				(*CUDA_LCC).dave[l] = 0;
		}
		else
			lave = (*CUDA_LCC).ave;
	}
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = (*CUDA_CC).ma / BLOCK_DIM;
	if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
	brtmpl++;

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	//if (threadIdx.x == 0)
	//	printf("conv>>> [%d] \n", blockIdx.x);

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		// *--- CONV() ---* //
		ymod = conv(CUDA_LCC, CUDA_CC, res, jp - 1, tmpl, tmph, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).ytemp[jp] = ymod;

			if (Inrel == 1)
				lave = lave + ymod;
		}
		for (l = tmpl; l <= tmph; l++)
		{
			//(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];
			(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];

			if (Inrel == 1)
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
		}
		/* save lightcurves */
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
					 (*CUDA_LCC).Yout[np] = ymod;*/
	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp;
		(*CUDA_LCC).ave = lave;
	}
}

double mrqcof_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha)
{
	int j, k;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	for (int j = 2; j <= (*CUDA_CC).Mfit; j++)
	{
		for (k = 1; k <= j - 1; k++)
		{
			alpha[k * (*CUDA_CC).Mfit1 + j] = alpha[j * (*CUDA_CC).Mfit1 + k];
			//if (blockIdx.x ==0 && threadIdx.x == 0)
			//	printf("[mrqcof_end] [%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, k * (*CUDA_CC).Mfit1 + j, alpha[k * (*CUDA_CC).Mfit1 + j]);
		}
	}

	return (*CUDA_LCC).trial_chisq;
}
