__kernel void Iter1Mrqmin1EndPre1(
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

	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / BLOCK_DIM;
	if (ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;

    if (!(*CUDA_LCC).isAlamda) return;
	
	for (j = tmpl; j <= tmph; j++)
	{
		(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

