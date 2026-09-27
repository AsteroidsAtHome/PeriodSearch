//Convexity regularization function

//  8.11.2006


double conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* res,
	int nc,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	double tmp = 0.0;
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

	/* the derivatives w.r.t. the shape coefficients are computed for all
	   points at once in mrqcof_curve1_last */

	return (tmp);
}
