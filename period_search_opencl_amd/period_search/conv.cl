//Convexity regularization function

//  8.11.2006

/* the last "lightcurve" holds the CONV_POINTS convexity constraint points
   (Lpoints == 3 for it, see period_search_BOINC.cpp); point jp uses column
   nc = jp - 1 of Nor */
#define CONV_POINTS 3

/* The derivatives w.r.t. the shape coefficients are computed for all points at
   once: Area[i] and Dsph[i][j] are read once per facet and shared by the three
   points instead of being re-read for every point. Every product and every
   ascending-i sum is formed exactly as in the old one-point-per-call version
   (Area[i] * Dsph[i][j] rounded first, then * Nor[i][nc]; the ymod reduction
   pairs the same elements), so the results are bit-identical.

   ymod[nc] is valid on work-item 0 only; dyda of point nc is written straight
   to dytemp row nc (and accumulated into dave for relative curves). */
void conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* res,   /* [CONV_POINTS * BLOCK_DIM] */
	int Lpoints,           /* <= CONV_POINTS */
	int Inrel,
	int tmpl,
	int tmph,
	int brtmpl,
	int brtmph,
	__global double* dytempG,
	double* ymod)          /* [CONV_POINTS] */
{
	int i, j, k, nc;
	double tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	for (i = brtmpl; i <= brtmph; i++)
	{
		double ar = (*CUDA_LCC).Area[i];
		tmp0 += ar * (*CUDA_CC).Nor[i][0];
		tmp1 += ar * (*CUDA_CC).Nor[i][1];
		tmp2 += ar * (*CUDA_CC).Nor[i][2];
	}

	res[threadIdx.x] = tmp0;
	res[BLOCK_DIM + threadIdx.x] = tmp1;
	res[2 * BLOCK_DIM + threadIdx.x] = tmp2;

	barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k)
		{
			res[threadIdx.x] += res[threadIdx.x + k];
			res[BLOCK_DIM + threadIdx.x] += res[BLOCK_DIM + threadIdx.x + k];
			res[2 * BLOCK_DIM + threadIdx.x] += res[2 * BLOCK_DIM + threadIdx.x + k];
		}
		k = k >> 1;
		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		for (nc = 0; nc < CONV_POINTS; nc++)
			ymod[nc] = res[nc * BLOCK_DIM] + res[nc * BLOCK_DIM + 1];
	}
	//parallel reduction end

	for (j = tmpl; j <= tmph; j++)
	{
		double dtmp[CONV_POINTS] = { 0, 0, 0 };
		if (j <= (*CUDA_CC).Ncoef)
		{
			for (i = 1; i <= (*CUDA_CC).Numfac; i++)
			{
				/* Darea[i] * Dg[i][j] == Area[i] * Dsph[i][j] (Area = Darea*g) */
				double ad = (*CUDA_LCC).Area[i] * (*CUDA_CC).Dsph[i][j];
				dtmp[0] += ad * (*CUDA_CC).Nor[i][0];
				dtmp[1] += ad * (*CUDA_CC).Nor[i][1];
				dtmp[2] += ad * (*CUDA_CC).Nor[i][2];
			}
		}

		for (nc = 0; nc < Lpoints; nc++)
		{
			dytempG[nc * DYT_STRIDE + j] = dtmp[nc];

			if (Inrel == 1)
				(*CUDA_LCC).dave[j] = (*CUDA_LCC).dave[j] + dtmp[nc];
		}
	}
}
