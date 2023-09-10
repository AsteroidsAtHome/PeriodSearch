
//int gauss_errc(freq_context* CUDA_LCC, const int ma)
//mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, block);
//int gauss_errc(struct mfreq_context* CUDA_LCC, struct freq_context* CUDA_CC, int* sh_icol, int* sh_irow, double* sh_big, int icol, double pivinv)
int gauss_errc(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC)
{
	//__shared__ int icol;
	//__shared__ double pivinv;
	//__shared__ int sh_icol[CUDA_BLOCK_DIM];
	//__shared__ int sh_irow[CUDA_BLOCK_DIM];
	//__shared__ double sh_big[CUDA_BLOCK_DIM];

	double big, dum, temp;
	double tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;
	int n = (*CUDA_CC).Mfit; // 54
	int m = (*CUDA_CC).ma;   // 57

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;		// 1 (thr 1)
	brtmpl = threadIdx.x * brtmph;		// 0
	brtmph = brtmpl + brtmph;			// 1
	if (brtmph > n) brtmph = n;			// false | 1
	brtmpl++;							// 1

	// <<< GausErrorCPre
	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) (*CUDA_LCC).ipiv[j] = 0;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// >>> GausErrorCPre End

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("brtmpl: %3d, brtmph: %3d\n", brtmpl, brtmph);

	// <<< GausErrorC
	for (i = 1; i <= n; i++)
	{
		big = 0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)  // 1 to 1 on thread 0 first pass for all "i"
		{
			//if (threadIdx.x == 0 && i == 2)
			//	printf("[%d][%3d] ipiv[%3d]: %5d, covar[%3d]: %10.7f\n",
			//		blockIdx.x, threadIdx.x, j, (*CUDA_LCC).ipiv[j], j * (*CUDA_CC).Mfit1 + 1, (*CUDA_LCC).covar[j * (*CUDA_CC).Mfit1 + 1]);

			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				//if (blockIdx.x == 0)
				//	printf("[%3d] i[%3d] ipiv[%3d]: %10.7f\n", threadIdx.x, i, j, (*CUDA_LCC).ipiv[j]);

				int ixx = j * (*CUDA_CC).Mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						double tmpcov = fabs((*CUDA_LCC).covar[ixx]);
						if (tmpcov >= big)
						{
							//if (blockIdx.x == 0)
							//	printf("[%3d] i[%3d] ipiv[%3d]: %3d, ipiv[%3d]: %3d, big: %10.7f, tmpcov: %10.7f, covar[%3d]: %10.7f\n",
							//		threadIdx.x, i, j, (*CUDA_LCC).ipiv[j], k, (*CUDA_LCC).ipiv[k], big, tmpcov, ixx, (*CUDA_LCC).covar[ixx]);

							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						//printf("-");
						barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
						/*					        deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
						return(1);
					}
				}
			}
		}
		(*CUDA_LCC).sh_big[threadIdx.x] = big;
		(*CUDA_LCC).sh_irow[threadIdx.x] = irow;
		(*CUDA_LCC).sh_icol[threadIdx.x] = licol;

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		//int d = (*CUDA_LCC).sh_icol[0];
		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[%3d][%3d] i: %3d, licol: %3d\n", blockIdx.x, threadIdx.x, i, licol);
		//	//printf("[%3d][%3d] i: %3d, sh_col[%3d]: %d, d: %3d\n", blockIdx.x, threadIdx.x, i, threadIdx.x, (*CUDA_LCC).sh_icol[threadIdx.x], d);

		if (threadIdx.x == 0)
		{
			big = (*CUDA_LCC).sh_big[0];				// = 0
			(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[0];	// = 0
			irow = (*CUDA_LCC).sh_irow[0];				// = 0

			for (j = 1; j < BLOCK_DIM; j++)				// 1..127
			{
				//if (blockIdx.x == 0 && i == 1)
				//	printf("sh_big[%3d]: %10.7f\n", j, (*CUDA_LCC).sh_big[j]);

				if ((*CUDA_LCC).sh_big[j] >= big)
				{
					big = (*CUDA_LCC).sh_big[j];
					irow = (*CUDA_LCC).sh_irow[j];
					(*CUDA_LCC).icol = (*CUDA_LCC).sh_icol[j];
				}
			}

			//(*CUDA_LCC).ipiv[(*CUDA_LCC).icol] = ++(*CUDA_LCC).ipiv[(*CUDA_LCC).icol];
			++(*CUDA_LCC).ipiv[(*CUDA_LCC).icol];

			//if (blockIdx.x == 0)
			//	printf("i: %2d, icol: %3d, irow: %3d, ipiv[%3d]: %3d\n", i, (*CUDA_LCC).icol, irow, (*CUDA_LCC).icol, (*CUDA_LCC).ipiv[(*CUDA_LCC).icol]);


			if (irow != (*CUDA_LCC).icol) // what is going on here ???
			{
				//if (blockIdx.x == 0)
				//	printf("irow: %3d\n", irow);
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

			//if (blockIdx.x == 0)
			//	printf("i: %3d, irow: %3d, icol: %3d\n", i, irow, (*CUDA_LCC).icol);

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

				return(2);
			}

			//<<<<<<<<<<  (*CUDA_LCC).
			(*CUDA_LCC).pivinv = 1.0 / (*CUDA_LCC).covar[covarIdx];
			(*CUDA_LCC).covar[covarIdx] = 1.0;


			(*CUDA_LCC).da[(*CUDA_LCC).icol] = (*CUDA_LCC).da[(*CUDA_LCC).icol] * (*CUDA_LCC).pivinv;
			//b[icol] *= pivinv;

			//if(blockIdx.x == 0)
			//	printf("[%d] i[%2d] da[%4d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).icol, (*CUDA_LCC).da[(*CUDA_LCC).icol]); // da - OK

			//if (blockIdx.x == 0)
			//	printf("[%d] i[%2d] pivinv: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).pivinv); // pivinv - OK

		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[%d] icol: %5d, mfit1: %3d, l: %3d\n", blockIdx.x, icol, (*CUDA_CC).Mfit1, l);

		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = (*CUDA_LCC).icol * (*CUDA_CC).Mfit1 + l;
			double covar1 = (*CUDA_LCC).covar[qq] * (*CUDA_LCC).pivinv;
			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("[%d][%3d] i[%3d] l[%3d] icol: %3d, pivinv: %10.7f, covar[%4d]: %10.7f, covar: %10.7f\n",
			//		blockIdx.x, threadIdx.x, i, l, (*CUDA_LCC).icol, (*CUDA_LCC).pivinv, qq, (*CUDA_LCC).covar[qq], covar1);

			//barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);// | CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

			//covar[qq] = 1.0;
			//(*CUDA_LCC).covar[qq] = (*CUDA_LCC).covar[qq] * pivinv;
			(*CUDA_LCC).covar[qq] = covar1;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("i[%d%3d] ll: %4d, brtmpl: %3d, brtmph; %3d\n", i, ll, brtmpl, brtmph);

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
				//b[ll] -= b[icol]*dum;

			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	// << GausErrorCPost
	if (threadIdx.x == 0)
	{
		for (l = n; l >= 1; l--)
		{
			if ((*CUDA_LCC).indxr[l] != (*CUDA_LCC).indxc[l])
			{
				for (k = 1; k <= n; k++)
				{
					//SwapDouble((*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]], (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]]);
					tmpSwap = (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]];
					(*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxr[l]] = (*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]];
					(*CUDA_LCC).covar[k * (*CUDA_CC).Mfit1 + (*CUDA_LCC).indxc[l]] = tmpSwap;
				}
			}
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return(0);
	// >>> GaussErrorCPost END
}
// #undef SWAP
 //from Numerical Recipes

