__kernel void GaussErrorCPre(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
    int j;
    int3 blockIdx;
	blockIdx.x = get_group_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
    
    int n = (*CUDA_CC).Mfit; // 54

    for (j = 1; j <= n; j++) 
    {
        (*CUDA_LCC).ipiv[j] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

