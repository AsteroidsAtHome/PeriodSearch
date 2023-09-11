__kernel void Iter1Mrqcof1Curve2NonRel(
    __global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
    const int lpoints)
{
    int l, j, jp, k, m, lnp2, Lpoints1 = lpoints + 1;
    double dy, ymod, sig2i, wt, wght, ltrial_chisq;
    int tmph, tmpl;

    int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;

    __global double* alpha = (*CUDA_LCC).alpha;
    __global double* beta = (*CUDA_LCC).beta;

    int matmph, matmpl;									// threadIdx.x == 1
    matmph = (*CUDA_CC).ma / BLOCK_DIM;					// 0
    if ((*CUDA_CC).ma % BLOCK_DIM) matmph++;			// 1
    matmpl = threadIdx.x * matmph;						// 1
    matmph = matmpl + matmph;							// 2
    if (matmph > (*CUDA_CC).ma) matmph = (*CUDA_CC).ma;
    matmpl++;	

    int latmph, latmpl;
    latmph = (*CUDA_CC).lastone / BLOCK_DIM;
    if ((*CUDA_CC).lastone % BLOCK_DIM) latmph++;
    latmpl = threadIdx.x * latmph;
    latmph = latmpl + latmph;
    if (latmph > (*CUDA_CC).lastone) latmph = (*CUDA_CC).lastone;
    latmpl++;

    lnp2 = (*CUDA_LCC).np2;						//  Include this two in both, NonRel & Rel
    ltrial_chisq = (*CUDA_LCC).trial_chisq;		//

    // if ((*CUDA_CC).ia[1]) //not relative
    // {
        for (jp = 1; jp <= lpoints; jp++)
        {
            ymod = (*CUDA_LCC).ytemp[jp];

            int ixx = jp + matmpl * Lpoints1;
            for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
                (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

            lnp2++;

            //xx = tex1Dfetch(texsig, lnp2);
            //sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));
            sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);

            //xx = tex1Dfetch(texWeight, lnp2);
            //wght = __hiloint2double(xx.y, xx.x);
            wght = (*CUDA_CC).Weight[lnp2];

            //xx = tex1Dfetch(texbrightness, lnp2);
            //dy = __hiloint2double(xx.y, xx.x) - ymod;
            dy = (*CUDA_CC).Brightness[lnp2] - ymod;

            j = 0;
            //
            double sig2iwght = sig2i * wght;
            //
            for (l = 1; l <= (*CUDA_CC).lastone; l++)
            {
                j++;
                wt = (*CUDA_LCC).dyda[l] * sig2iwght;
                //				   k = 0;
                //precalc thread boundaries
                tmph = l / BLOCK_DIM;
                if (l % BLOCK_DIM) tmph++;
                tmpl = threadIdx.x * tmph;
                tmph = tmpl + tmph;
                if (tmph > l) tmph = l;
                tmpl++;
                for (m = tmpl; m <= tmph; m++)
                {
                    //				  k++;
                    alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + wt * (*CUDA_LCC).dyda[m];
                } /* m */
                barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                if (threadIdx.x == 0)
                {
                    beta[j] = beta[j] + dy * wt;
                }
                barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
            } /* l */
            for (; l <= (*CUDA_CC).lastma; l++)
            {
                if ((*CUDA_CC).ia[l])
                {
                    j++;
                    wt = (*CUDA_LCC).dyda[l] * sig2iwght;
                    //				   k = 0;

                    for (m = latmpl; m <= latmph; m++)
                    {
                        //					  k++;
                        alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + wt * (*CUDA_LCC).dyda[m];
                    } /* m */
                    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                    if (threadIdx.x == 0)
                    {
                        k = (*CUDA_CC).lastone;
                        m = (*CUDA_CC).lastone + 1;
                        for (; m <= l; m++)
                        {
                            if ((*CUDA_CC).ia[m])
                            {
                                k++;
                                alpha[j * (*CUDA_CC).Mfit1 + k] = alpha[j * (*CUDA_CC).Mfit1 + k] + wt * (*CUDA_LCC).dyda[m];
                            }
                        } /* m */
                        beta[j] = beta[j] + dy * wt;
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                }
            } /* l */
            ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
        } /* jp */
    // }

    if (threadIdx.x == 0)
    {
        (*CUDA_LCC).np2 = lnp2;
        (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
}
