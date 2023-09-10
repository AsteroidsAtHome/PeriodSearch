__kernel void Iter1Mrqmin1EndPost(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
    __global int* ErrCode)
{
    int l, j = 0;
    int3 blockIdx;
	blockIdx.x = get_group_id(0);

    int ma = (*CUDA_CC).ma;

    if (ErrCode[blockIdx.x]) return;

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

    // if (threadIdx.x == 0)
	// {
		//		if (err_code != 0) return(err_code);  "bacha na sync threads" - Watch out for Sync Threads
		// j = 0;
		for (int l = 1; l <= ma; l++)
			if ((*CUDA_CC).ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
			}
	// }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

