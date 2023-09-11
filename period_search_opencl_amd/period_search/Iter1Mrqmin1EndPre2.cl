__kernel void Iter1Mrqmin1EndPre2(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
    int j;

    int3 threadIdx, blockIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

    int ma = (*CUDA_CC).ma;

	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * (*CUDA_CC).Mfit1 + 1;
		for (int k = 1; k <= (*CUDA_CC).Mfit; k++, ixx++)
		{
			(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];
		}

		int qq = j * (*CUDA_CC).Mfit1 + j;
		(*CUDA_LCC).covar[qq] = (*CUDA_LCC).alpha[qq] * (1 + (*CUDA_LCC).Alamda);
		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

