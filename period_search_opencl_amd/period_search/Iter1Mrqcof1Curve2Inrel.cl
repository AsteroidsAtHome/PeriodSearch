__kernel void Iter1Mrqcof1Curve2Inrel(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int lpoints)
{
    int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;

    __global double* alpha = (*CUDA_LCC).alpha;
    __global double* beta = (*CUDA_LCC).beta;

    int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
    double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;

    //precalc thread boundaries
    int tmph, tmpl;
    tmph = lpoints / BLOCK_DIM;
    if (lpoints % BLOCK_DIM) tmph++;
    tmpl = threadIdx.x * tmph;
    lnp1 = (*CUDA_LCC).np1 + tmpl;
    tmph = tmpl + tmph;
    if (tmph > lpoints) tmph = lpoints;
    tmpl++;

    int matmph, matmpl;									// threadIdx.x == 1
    matmph = (*CUDA_CC).ma / BLOCK_DIM;					// 0
    if ((*CUDA_CC).ma % BLOCK_DIM) matmph++;			// 1
    matmpl = threadIdx.x * matmph;						// 1
    matmph = matmpl + matmph;							// 2
    if (matmph > (*CUDA_CC).ma) matmph = (*CUDA_CC).ma;
    matmpl++;											// 2

    int latmph, latmpl;
    latmph = (*CUDA_CC).lastone / BLOCK_DIM;
    if ((*CUDA_CC).lastone % BLOCK_DIM) latmph++;
    latmpl = threadIdx.x * latmph;
    latmph = latmpl + latmph;
    if (latmph > (*CUDA_CC).lastone) latmph = (*CUDA_CC).lastone;
    latmpl++;

    /*   if ((*CUDA_LCC).Lastcall != 1) always ==0
         {*/
    // if (inrel /*==1*/)
    // {
        for (jp = tmpl; jp <= tmph; jp++)
        {
            lnp1++;
            int ixx = jp + 1 * Lpoints1;

            /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
            (*CUDA_LCC).dytemp[ixx] = 0;

            coef = (*CUDA_CC).Sig[lnp1] * lpoints / (*CUDA_LCC).ave;

            double yytmp = (*CUDA_LCC).ytemp[jp];
            coef1 = yytmp / (*CUDA_LCC).ave;

            (*CUDA_LCC).ytemp[jp] = coef * yytmp;
            ixx += Lpoints1;

            for (l = 2; l <= (*CUDA_CC).ma; l++, ixx += Lpoints1)
            {
                (*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);
            }
        }
    // }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();
}
