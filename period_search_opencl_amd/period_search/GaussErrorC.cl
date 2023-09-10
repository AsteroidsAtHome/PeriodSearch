__kernel void GaussErrorC(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
    __global int* ErrCode,
    int n)
{
    double big, dum, temp;
	double tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;

    int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

    ErrCode[blockIdx.x] = 0;
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

    int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;		// 1 (thr 1)
	brtmpl = threadIdx.x * brtmph;		// 0
	brtmph = brtmpl + brtmph;			// 1
	if (brtmph > n) brtmph = n;			// false | 1
	brtmpl++;							// 1

    // for (i = 1; i <= n; i++)
	// {
        
		big = 0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)  // 1 to 1 on thread 0 first pass for all "i"
		{
			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				int ixx = j * (*CUDA_CC).Mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						double tmpcov = fabs((*CUDA_LCC).covar[ixx]);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                        ErrCode[blockIdx.x] = 1;
						// return(1);
						return;
					}
				}
			}
		}
		(*CUDA_LCC).sh_big[threadIdx.x] = big;
		(*CUDA_LCC).sh_irow[threadIdx.x] = irow;
		(*CUDA_LCC).sh_icol[threadIdx.x] = licol;

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (threadIdx.x == 0)
		{
			big = (*CUDA_LCC).sh_big[0];				// = 0
			(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[0];	// = 0
			irow = (*CUDA_LCC).sh_irow[0];				// = 0

			for (j = 1; j < BLOCK_DIM; j++)				// 1..127
			{
				if ((*CUDA_LCC).sh_big[j] >= big)
				{
					big = (*CUDA_LCC).sh_big[j];
					irow = (*CUDA_LCC).sh_irow[j];
					(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[j];
				}
			}

			++(*CUDA_LCC).ipiv[(*CUDA_LCC).icol];

			if (irow != (*CUDA_LCC).icol) // what is going on here ???
			{
				for (l = 1; l <= n; l++)
				{
					//SwapDouble((*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l], (*CUDA_LCC).covar[icol * (*CUDA_CC).Mfit1 + l]);
					tmpSwap = (*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l];
					(*CUDA_LCC).covar[irow * (*CUDA_CC).Mfit1 + l] = (*CUDA_LCC).covar[(*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l];
					(*CUDA_LCC).covar[(*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l] = tmpSwap;

				}

				//SwapDouble((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol]);
				tmpSwap = (*CUDA_LCC).da[irow];
				(*CUDA_LCC).da[irow] = (*CUDA_LCC).da[(*CUDA_LCC).icol];
				(*CUDA_LCC).da[(*CUDA_LCC).icol] = tmpSwap;

				//SWAP(b[irow],b[icol])
			}

			(*CUDA_LCC).indxr[i] = irow;
			(*CUDA_LCC).indxc[i] = (*CUDA_LCC).icol;

			int covarIdx = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + (*CUDA_LCC).icol;
			if ((*CUDA_LCC).covar[covarIdx] == 0.0)
			{
				j = 0;
				for (int l = 1; l <= (*CUDA_CC).ma; l++)
				{
					if ((*CUDA_CC).ia[l])
					{
						j++;
						(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
					}
				}

                ErrCode[blockIdx.x] = 2;
				// return(2);
				return;
			}

			//<<<<<<<<<<  (*CUDA_LCC).
			(*CUDA_LCC).pivinv = 1.0 / (*CUDA_LCC).covar[covarIdx];
			(*CUDA_LCC).covar[covarIdx] = 1.0;
			(*CUDA_LCC).da[(*CUDA_LCC).icol] = (*CUDA_LCC).da[(*CUDA_LCC).icol] * (*CUDA_LCC).pivinv;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l;
			(*CUDA_LCC).covar[qq] = (*CUDA_LCC).covar[qq] * (*CUDA_LCC).pivinv;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			if (ll != (*CUDA_LCC).icol)
			{
				int ixx = ll * (*CUDA_CC).Mfit1;
				int jxx = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1;
				dum = (*CUDA_LCC).covar[ixx + (*CUDA_LCC).icol];
				(*CUDA_LCC).covar[ixx + (*CUDA_LCC).icol] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				}

				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[(*CUDA_LCC).icol] * dum;
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// }
}

