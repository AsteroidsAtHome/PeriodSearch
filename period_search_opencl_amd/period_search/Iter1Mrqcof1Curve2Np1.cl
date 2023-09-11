__kernel void Iter1Mrqcof1Curve2Np1(
    __global struct mfreq_context* CUDA_mCC,
    const int lpoints)
{
    int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;

    (*CUDA_LCC).np1 += lpoints;
}
