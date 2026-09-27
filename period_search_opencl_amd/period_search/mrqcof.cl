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


	//int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%d][%3d] cg[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, q, (*CUDA_LCC).cg[q]);


}

void mrqcof_matrix(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int Lpoints,
	int num,
	__global double* scr)
{
	matrix_neo(CUDA_LCC, CUDA_CC, cg, (*CUDA_LCC).np, Lpoints, num, scr);
}

void mrqcof_curve1(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__local double* tmave,
	int Inrel,
	int Lpoints,
	int num,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
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

	/* points are dealt out round-robin (jp = t+1, t+1+BLOCK_DIM, ...) rather than
	   in contiguous blocks of ceil(Lpoints/BLOCK_DIM): every point is
	   independent, and this packs the last partial round into as few
	   wavefronts as possible (156 points: 5 wave32 rounds instead of 6). The
	   ytemp partial sums below keep the contiguous blocks, so ave is summed
	   in the same order as before. */
	for (jp = threadIdx.x + 1; jp <= Lpoints; jp += BLOCK_DIM)
	{
			/*  ---  BRIGHT  ---  */
		bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel, scr);
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
		for (int l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx = l;
			(*CUDA_LCC).dave[l] = dytempG[ixx];

			//jp>=2
			ixx += DYT_STRIDE;
			for (int jp = 2; jp <= Lpoints; jp++, ixx += DYT_STRIDE)
			{
				//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dytempG[ixx];
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dytempG[ixx];

				//if (threadIdx.x == 1)
				//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, dytempG[ixx], l, (*CUDA_LCC).dave[l]);
			}
		}

		tmave[threadIdx.x] = 0;
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += ytempG[jp];
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
	int Lpoints,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
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

	/* convexity derivatives dyda[l] = sum_i Area[i] * Dsph[i][l] * Nor[i][nc]
	   of every point (nc = jp-1) in ONE pass over the facets - it used to be
	   one pass per point inside conv(). Area/Dsph are read once for all
	   points and the per-point sums become independent chains; each sum keeps
	   its facet order and operand rounding, and dave[l] still accumulates the
	   points in ascending order (each l belongs to one work-item). */
	for (l = tmpl; l <= tmph; l++)
	{
		for (int jb = 0; jb < Lpoints; jb += 3)
		{
			double d[3] = { 0, 0, 0 };
			if (l <= (*CUDA_CC).Ncoef)
			{
				for (int i = 1; i <= (*CUDA_CC).Numfac; i++)
				{
					/* Darea[i] * Dg[i][l] == Area[i] * Dsph[i][l] (Area = Darea*g) */
					double ad = (*CUDA_LCC).Area[i] * (*CUDA_CC).Dsph[i][l];
					for (int q = 0; q < 3; q++)
						d[q] += ad * (*CUDA_CC).Nor[i][jb + q];
				}
			}
			for (int q = 0; q < 3 && jb + q < Lpoints; q++)
			{
				dytempG[(jb + q) * DYT_STRIDE + l] = d[q];
				if (Inrel == 1)
					(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + d[q];
			}
		}
	}

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		// *--- CONV() ---* //
		ymod = conv(CUDA_LCC, CUDA_CC, res, jp - 1, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			ytempG[jp] = ymod;

			if (Inrel == 1)
				lave = lave + ymod;
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

	/* mirror the lower triangle; each row is split over the work-group
	   (reads are contiguous, source and destination never overlap) */
	int lsize = get_local_size(0);
	for (int j = 2; j <= (*CUDA_CC).Mfit; j++)
	{
		for (k = 1 + threadIdx.x; k <= j - 1; k += lsize)
		{
			alpha[k * (*CUDA_CC).Mfit1 + j] = alpha[j * (*CUDA_CC).Mfit1 + k];
			//if (blockIdx.x ==0 && threadIdx.x == 0)
			//	printf("[mrqcof_end] [%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, k * (*CUDA_CC).Mfit1 + j, alpha[k * (*CUDA_CC).Mfit1 + j]);
		}
	}

	return (*CUDA_LCC).trial_chisq;
}
