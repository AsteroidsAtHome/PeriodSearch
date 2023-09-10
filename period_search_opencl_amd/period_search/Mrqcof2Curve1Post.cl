__kernel void Mrqcof2Curve1Post(
    __global struct mfreq_context* CUDA_mCC,
	int Lpoints)
{
    int3 blockIdx;
	blockIdx.x = get_group_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;    

	// __private int lnp = (*CUDA_LCC).np;

	(*CUDA_LCC).np += Lpoints;
	// (*CUDA_LCC).ave = lave;

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
}

