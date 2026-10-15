__kernel void GaussErrorCPost(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
    __global int* ErrCode)
{
    double tmpSwap;
    int l, k;
    int3 blockIdx;
	blockIdx.x = get_group_id(0);
    int n = (*CUDA_CC).Mfit; // 54

    if (ErrCode[blockIdx.x]) return;

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    
    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

    // if (threadIdx.x == 0)
	// {
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

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	// }  
}

