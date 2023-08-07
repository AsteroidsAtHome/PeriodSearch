//Convexity regularization function

//  8.11.2006


double conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* res,
	int nc,
	int tmpl,
	int tmph,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	double tmp = 0.0;
	double dtmp;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//j = blockIdx.x * (CUDA_Numfac1)+brtmpl;
	j = brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		//tmp += CUDA_Area[j] * CUDA_Nor[i][nc];
		tmp += (*CUDA_LCC).Area[j] * (*CUDA_CC).Nor[i][nc];
	}

	res[threadIdx.x] = tmp;

	//if (threadIdx.x == 0)
	//    printf("conv>>> [%d] jp-1[%3d] res[%3d]: %10.7f\n", blockIdx.x, nc, threadIdx.x, res[threadIdx.x]);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k)
			res[threadIdx.x] += res[threadIdx.x + k];
		k = k >> 1;
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		tmp = res[0] + res[1];
	}
	//parallel reduction end
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//int m = blockIdx.x * (*CUDA_CC).Dg_block + tmpl * (*CUDA_CC).Numfac1);   // <<<<<<<<<<<<<<<<<<<<<<<<<<<<< !!!
	int m = tmpl * (*CUDA_CC).Numfac1;
	for (j = tmpl; j <= tmph; j++)  //, m += (*CUDA_CC).Numfac1)
	{
		// printf("m: %4d\n", m);
		dtmp = 0;
		if (j <= (*CUDA_CC).Ncoef)
		{
			int mm = m + 1;
			for (i = 1; i <= (*CUDA_CC).Numfac; i++, mm++)
			{
				// dtmp += CUDA_Darea[i] * CUDA_Dg[mm] * CUDA_Nor[i][nc];
				dtmp += (*CUDA_CC).Darea[i] * (*CUDA_LCC).Dg[mm] * (*CUDA_CC).Nor[i][nc];

				//if (blockIdx.x == 0 && j == 8)
				//	printf("[%d][%3d]  Darea[%4d]: %.7f, Dg[%4d]: %.7f, Nor[%3d][%3d]: %10.7f\n",
				//		blockIdx.x, threadIdx.x, i, (*CUDA_CC).Darea[i], mm, (*CUDA_LCC).Dg[mm], i, nc, (*CUDA_CC).Nor[i][nc]);
			}
		}

		(*CUDA_LCC).dyda[j] = dtmp;

		//if (blockIdx.x == 0) // && threadIdx.x == 1)
		//    printf("[mrqcof_curve1_last -> conv] [%d][%3d] jp - 1: %3d, j[%3d] dyda[%3d]: %10.7f\n",
		//        blockIdx.x, threadIdx.x, nc, j, j, (*CUDA_LCC).dyda[j]);
	}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return (tmp);
}
