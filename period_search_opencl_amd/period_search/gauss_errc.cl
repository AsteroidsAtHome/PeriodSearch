
//from Numerical Recipes

/* 2026: the damped normal matrix is staged into local memory and the whole
   Gauss-Jordan elimination runs there; global memory is only touched to read
   alpha/beta on entry and to write the step vector da at the end. The old
   version swept covar in global memory on every pivot step. Two consequences
   of the caller's structure are used:

   * the inverted matrix itself is dead - ClCalculateIter1Mrqcof2Start rezeroes
	 covar before mrqcof2 accumulates into it, and mrqmin_2_end copies that
	 fresh accumulation - so neither the solved matrix nor the final
	 column-unscramble pass (and its indxr/indxc bookkeeping) is needed;
	 only da and the return code leave this function;

   * the icol/pivinv broadcast scalars and the pivot-reduction arrays move
	 from per-context global struct members to local memory.

   The local buffers are declared at kernel scope (OpenCL requirement) in
   ClCalculateIter1Mrqmin1End and passed through mrqmin_1_end. Pivot choice
   and elimination order are unchanged, so the computed step is bit-identical
   to the global-memory version. */
int gauss_errc(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* covL,   /* [DYT_STRIDE * DYT_STRIDE], indexed with Mfit1 stride */
	__local double* daL,    /* [DYT_STRIDE] */
	__local int* ipivL,     /* [DYT_STRIDE] */
	__local double* shBig,  /* [BLOCK_DIM] */
	__local int* shIrow,    /* [BLOCK_DIM] */
	__local int* shIcol,    /* [BLOCK_DIM] */
	__local double* pivBC,  /* [1] pivinv broadcast */
	__local int* icolBC,    /* [1] icol broadcast */
	__global double* alphaG)
{
	double big, dum;
	double tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;
	int n = (*CUDA_CC).Mfit;
	int mfit1 = (*CUDA_CC).Mfit1;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > n) brtmph = n;
	brtmpl++;

	/* stage the damped matrix and the right-hand side straight from
	   alpha/beta (this replaces the covar staging that mrqmin_1_end used to
	   do in global memory; covar itself is no longer written at all) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * mfit1 + 1;
		for (k = 1; k <= n; k++, ixx++)
		{
			covL[ixx] = alphaG[ixx];
		}
		int qq = j * mfit1 + j;
		covL[qq] = alphaG[qq] * (1 + (*CUDA_LCC).Alamda);
		daL[j] = (*CUDA_LCC).beta[j];
	}

	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) ipivL[j] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

	for (i = 1; i <= n; i++)
	{
		big = 0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
		{
			if (ipivL[j] != 1)
			{
				int ixx = j * mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if (ipivL[k] == 0)
					{
						double tmpcov = fabs(covL[ixx]);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if (ipivL[k] > 1)
					{
						barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
						return(1);
					}
				}
			}
		}
		shBig[threadIdx.x] = big;
		shIrow[threadIdx.x] = irow;
		shIcol[threadIdx.x] = licol;

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (threadIdx.x == 0)
		{
			big = shBig[0];
			icolBC[0] = shIcol[0];
			irow = shIrow[0];

			for (j = 1; j < BLOCK_DIM; j++)
			{
				if (shBig[j] >= big)
				{
					big = shBig[j];
					irow = shIrow[j];
					icolBC[0] = shIcol[j];
				}
			}

			++ipivL[icolBC[0]];

			if (irow != icolBC[0])
			{
				for (l = 1; l <= n; l++)
				{
					tmpSwap = covL[irow * mfit1 + l];
					covL[irow * mfit1 + l] = covL[icolBC[0] * mfit1 + l];
					covL[icolBC[0] * mfit1 + l] = tmpSwap;
				}

				tmpSwap = daL[irow];
				daL[irow] = daL[icolBC[0]];
				daL[icolBC[0]] = tmpSwap;
			}

			int covarIdx = icolBC[0] * mfit1 + icolBC[0];

			if (covL[covarIdx] == 0.0)
			{
				/* singular pivot: report the (partial) step like the old code
				   did, then bail with error 2 */
				for (j = 1; j <= n; j++)
				{
					(*CUDA_LCC).da[j] = daL[j];
				}
				j = 0;
				for (int l2 = 1; l2 <= (*CUDA_CC).ma; l2++)
				{
					if ((*CUDA_CC).ia[l2])
					{
						j++;
						(*CUDA_LCC).atry[l2] = (*CUDA_LCC).cg[l2] + (*CUDA_LCC).da[j];
					}
				}

				return(2);
			}

			pivBC[0] = ddiv(1.0, covL[covarIdx]);
			covL[covarIdx] = 1.0;

			daL[icolBC[0]] = daL[icolBC[0]] * pivBC[0];
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = icolBC[0] * mfit1 + l;
			double covar1 = covL[qq] * pivBC[0];
			covL[qq] = covar1;
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			if (ll != icolBC[0])
			{
				int ixx = ll * mfit1;
				int jxx = icolBC[0] * mfit1;
				dum = covL[ixx + icolBC[0]];
				covL[ixx + icolBC[0]] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					covL[ixx] -= covL[jxx] * dum;
				}

				daL[ll] -= daL[icolBC[0]] * dum;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	/* only the step vector leaves the solver (the column unscramble of the
	   classic routine acted on the inverse, which nothing reads) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		(*CUDA_LCC).da[j] = daL[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return(0);
}
