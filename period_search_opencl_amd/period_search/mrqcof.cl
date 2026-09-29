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

	if (Inrel == 1)
	{
		/* the sums below read other work-items' dytemp/ytemp rows */
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

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
	__local double* res,   /* [CONV_POINTS * BLOCK_DIM] */
	int Inrel,
	int Lpoints,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	int l, jp, lnp;
	double ymod[CONV_POINTS], lave;
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
	//if (threadIdx.x == 0)
	//	printf("conv>>> [%d] \n", blockIdx.x);

	/* the derivatives w.r.t. the shape coefficients are computed for all
	   points at once (see conv.cl); conv's first barrier also orders the dave
	   reset above before its dave updates */
	conv(CUDA_LCC, CUDA_CC, res, Lpoints, Inrel, tmpl, tmph, brtmpl, brtmph, dytempG, ymod);

	if (threadIdx.x == 0)
	{
		for (jp = 1; jp <= Lpoints; jp++)
		{
			ytempG[jp] = ymod[jp - 1];

			if (Inrel == 1)
				lave = lave + ymod[jp - 1];
		}
	}
	lnp += Lpoints;

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
	/* mirror the lower triangle into the upper one: alpha[k][j] = alpha[j][k]
	   for k < j. The work-items of the context split the (j, k) pairs; reads
	   (row > col) and writes (row < col) never overlap, so no ordering is
	   needed and every entry gets exactly the value it got before. */
	const int mfit = (*CUDA_CC).Mfit;
	const int mfit1 = (*CUDA_CC).Mfit1;
	const int total = mfit * (mfit - 1) / 2;

	/* pair e -> row j = 2.., column k = 1..j-1 (row-major lower triangle) */
	int j = 2, k = get_local_id(0) + 1;
	while (k > j - 1) { k -= j - 1; j++; }
	for (int e = get_local_id(0); e < total; e += BLOCK_DIM)
	{
		alpha[k * mfit1 + j] = alpha[j * mfit1 + k];

		k += BLOCK_DIM;
		while (k > j - 1) { k -= j - 1; j++; }
	}

	return (*CUDA_LCC).trial_chisq;
}
