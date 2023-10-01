kernel void ClCheckEnd(
	__global int* CUDA_End,
	int theEnd)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	if (blockIdx.x == 0)
		*CUDA_End = theEnd;

	//if (blockIdx.x == 0)
		//printf("CheckEnd CUDA_End: %2d\n", *CUDA_End);

}

__kernel void ClCalculatePrepare(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	double freq_start,
	double freq_step,
	int n_max,
	int n_start)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	int n = n_start + blockIdx.x;
	int isInvalid = 0;
	
	//zero context
	// Take care when dealing with branches. Particularly, avoid data loads and stores within the statements
	if (n > n_max)	{
		isInvalid = 1;
		// return;
	}	else	{
		isInvalid = 0;
	}

	(*CUDA_LCC).isInvalid = isInvalid;
	(*CUDA_FR).isInvalid = isInvalid;
	if(isInvalid == 1) return;

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

	///* initial poles */
	(*CUDA_LFR).per_best = 0.0;
	(*CUDA_LFR).dark_best = 0.0;
	(*CUDA_LFR).la_best = 0.0;
	(*CUDA_LFR).be_best = 0.0;
	(*CUDA_LFR).dev_best = 1e40;
}

__kernel void ClCalculatePreparePole(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global double* CUDA_cg_first,
	__global int* CUDA_End,
    __global struct freq_context* CUDA_CC2,
	int m)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	int isReported = (*CUDA_FR).isReported;
	int addend = 0;
	if ((*CUDA_LCC).isInvalid)
	{
		addend = 1;
		isReported = 0; //signal not to read result
	}

	atomic_add(CUDA_End, addend);
	(*CUDA_FR).isReported = 0; 
	if((*CUDA_LCC).isInvalid) return;

	double period = 1.0 / (*CUDA_LCC).freq;

	//* starts from the initial ellipsoid */
	for (int i = 1; i <= (*CUDA_CC).Ncoef; i++)
	{
		(*CUDA_LCC).cg[i] = CUDA_cg_first[i];
	}

	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = (*CUDA_CC).beta_pole[m];
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = (*CUDA_CC).lambda_pole[m];

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = 90.0 - (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];

	/* conversion of lambda, beta to radians */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];

	/* Use omega instead of period */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3] = 24.0 * 2.0 * PI / period;

	for (int i = 1; i <= (*CUDA_CC).Nphpar; i++)
	{
		(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i] = (*CUDA_CC).par[i];
	}

	/* Lommel-Seeliger part */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2] = 1;

	/* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = log((*CUDA_CC).cl);

	/* Levenberg-Marquardt loop */
	// moved to global iter_max,iter_min,iter_dif_max
	//
	(*CUDA_LCC).rchisq = -1;
	(*CUDA_LCC).Alamda = -1;
	(*CUDA_LCC).Niter = 0;
	(*CUDA_LCC).iter_diff = 1e40;
	(*CUDA_LCC).dev_old = 1e30;
	(*CUDA_LCC).dev_new = 0;
	//	(*CUDA_LCC).Lastcall=0; always ==0
	(*CUDA_LFR).isReported = 0;

    // if(blockIdx.x == 0)
    // {
    //     for(int i = 0; i < MAX_N_OBS + 1; i++)
    //     {
    //         (*CUDA_CC2).Brightness[i] = (*CUDA_CC).Brightness[i];
    //     }
    // }
}

__kernel void ClCalculateIter1Begin(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	int CUDA_n_iter_min,
	int CUDA_n_iter_max,
	double CUDA_iter_diff_max,
	double CUDA_Alamda_start)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	(*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);
	(*CUDA_FR).isNiter = (*CUDA_LCC).isNiter;

	int isAlamda = (*CUDA_LCC).isAlamda;
	double Alamda = (*CUDA_LCC).Alamda;
	int addend = 0;
	int isReported = (*CUDA_LFR).isReported;

	if ((*CUDA_LCC).isNiter){
		if ((*CUDA_LCC).Alamda < 0) {
			isAlamda = 1;
			Alamda = CUDA_Alamda_start; /* initial alambda */
		} else {
			isAlamda = 0;
		}

	} else {
		if (!(*CUDA_LFR).isReported){
			addend = 1;
			isReported = 1;
		}
	}

	(*CUDA_LCC).isAlamda = isAlamda;
	(*CUDA_LCC).Alamda = Alamda;
	atomic_add(CUDA_End, addend);
	(*CUDA_LFR).isReported = isReported;
}

__kernel void ClCalculateIter1Mrqcof1Start(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
	//__global int* CUDA_End)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return; //>> 0

	mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
}

__kernel void ClCalculateIter1Mrqcof1Matrix(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	int x = blockIdx.x;

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local int num; // __shared__

	int3 localIdx;
	localIdx.x = get_local_id(0);
	if (localIdx.x == 0)
	{
		num = 0;
	}

	mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, lpoints, num);
}

__kernel void ClCalculateIter1Mrqcof1Curve1(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	//double* dytemp = &CUDA_Dytemp[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local int num;  // __shared__
	__local double tmave[BLOCK_DIM];

	if (threadIdx.x == 0)
	{
		num = 0;
	}

	mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, tmave, inrel, lpoints, num);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("[Mrqcof1Curve1] [%d][%3d] alpha[56]: %10.7f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).alpha[56]);

	//if (blockIdx.x == 0)
	//	printf("dytemp[8636]: %10.7f\n", dytemp[8636]);
}

__kernel void ClCalculateIter1Mrqcof1Curve1Last(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	//double* dytemp = &CUDA_Dytemp[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local double res[BLOCK_DIM];

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof1Curve1Last\n");

	mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, res, inrel, lpoints);
	//if (threadIdx.x == 0)
	//{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++) {
	//		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	//}
}

__kernel void ClCalculateIter1Mrqcof1Curve2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;

	//mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
    __global double* alpha = (*CUDA_LCC).alpha;
    __global double* beta = (*CUDA_LCC).beta;

    int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
    double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;

    /*int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);*/


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
    if (inrel /*==1*/)
    {
        for (jp = tmpl; jp <= tmph; jp++)
        {
            lnp1++;
            int ixx = jp + 1 * Lpoints1;
            /* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
            (*CUDA_LCC).dytemp[ixx] = 0;

            //if (blockIdx.x == 0)
            //	printf("[%d][%d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);

            coef = (*CUDA_CC).Sig[lnp1] * lpoints / (*CUDA_LCC).ave;

            //if (threadIdx.x == 0)
            //	printf("[%d][%3d][%d] coef: %10.7f\n", blockIdx.x, threadIdx.x, jp, coef);

            double yytmp = (*CUDA_LCC).ytemp[jp];
            coef1 = yytmp / (*CUDA_LCC).ave;

            //if (blockIdx.x == 0 && threadIdx.x == 0)
            //	printf("[Device | mrqcof_curve2_1] [%3d]  yytmp[%3d]: %10.7f, ave: %10.7f\n", threadIdx.x, jp, yytmp, (*CUDA_LCC).ave);

            (*CUDA_LCC).ytemp[jp] = coef * yytmp;

            //if (blockIdx.x == 0)
            //	printf("[Device][%d][%3d] ytemp[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).ytemp[jp]);

            ixx += Lpoints1;

            //if (threadIdx.x == 0)
            //	printf("[%3d] jp[%3d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);

            for (l = 2; l <= (*CUDA_CC).ma; l++, ixx += Lpoints1)
            {
                (*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);

                //if (blockIdx.x == 0 && threadIdx.x == 0)
                //	printf("[Device | mrqcof_curve2_1] [%3d]  coef1: %10.7f, dave[%3d]: %10.7f, dytemp[%3d]: %10.7f\n",
                //		threadIdx.x, coef1, l, (*CUDA_LCC).dave[l], ixx, (*CUDA_LCC).dytemp[ixx]);
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();
	// <<< Iter1Mrqcof1Curve2Inrel END

	// >>> Iter1Mrqcof1Curve2Np1 | global: CUDA_grid_dim_precalc, local: 1
    if (threadIdx.x == 0)
    {
        (*CUDA_LCC).np1 += lpoints;
    }
	// <<< Iter1Mrqcof1Curve2Np1 END

	// <<< Iter1Mrqcof1Curve2NonRel
    lnp2 = (*CUDA_LCC).np2;						//  Include this two in both, NonRel & Rel
    ltrial_chisq = (*CUDA_LCC).trial_chisq;		//

    if ((*CUDA_CC).ia[1]) //not relative
    {
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
    }  // <<< Iter1Mrqcof1Curve2NonRel 
    else //relative ia[1]==0
    {  // >>> Iter1Mrqcof1Curve2Rel

        for (jp = 1; jp <= lpoints; jp++)
        {
            ymod = (*CUDA_LCC).ytemp[jp];

            int ixx = jp + matmpl * Lpoints1;
            for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
            {
                (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];  // jp[1] dytemp[315] 0.0 - ?!?  must be -1051420.6747227

            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

            lnp2++;
            sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);

            wght = (*CUDA_CC).Weight[lnp2];

            dy = (*CUDA_CC).Brightness[lnp2] - ymod;

            j = 0;
            //
            double sig2iwght = sig2i * wght;
            //l==1
            //
            for (l = 2; l <= (*CUDA_CC).lastone; l++)
            {

                j++;
                wt = (*CUDA_LCC).dyda[l] * sig2iwght; // jp[1]  dyda[2] == 0    - ?!? must be -1051420.6747227   *) See dytemp[]
                // jp 2, dyda[9] == 0 - ?!? must be 7.9447669

				//				   k = 0;
				//precalc thread boundaries
                tmph = l / BLOCK_DIM;
                if (l % BLOCK_DIM) tmph++;
                tmpl = threadIdx.x * tmph;
                tmph = tmpl + tmph;
                if (tmph > l) tmph = l;
                tmpl++;
                //m==1
                if (tmpl == 1) tmpl++;
                //
                for (m = tmpl; m <= tmph; m++)
                {
                    //					  k++;
                    alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];
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

                    tmpl = latmpl;
                    //m==1
                    if (tmpl == 1) tmpl++;
                    //
                    for (m = tmpl; m <= latmph; m++)
                    {
                        //k++;
                        alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];
                    } /* m */
                    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                    if (threadIdx.x == 0)
                    {
                        k = (*CUDA_CC).lastone - 1;
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
    }
    //     } always ==0 // Lastcall != 1

    if (threadIdx.x == 0)
    {
        (*CUDA_LCC).np2 = lnp2;
        (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
}

__kernel void ClCalculateIter1Mrqcof1End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof1End\n");


	(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha);


	////if (threadIdx.x == 0)
	////{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++) {
	//	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	////}
}

__kernel void ClCalculateIter1Mrqmin1End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

	mrqmin_1_end(CUDA_LCC, CUDA_CC);
}

__kernel void ClCalculateIter1Mrqcof2Start(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof2Start\n");


	//mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
	mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("alpha[56]: %10.7f\n", (*CUDA_LCC).alpha[56]);
}

__kernel void ClCalculateIter1Mrqcof2Matrix(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local int num; // __shared__

	int3 localIdx;
	localIdx.x = get_local_id(0);
	if (localIdx.x == 0)
	{
		num = 0;
	}

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof2Matrix\n");

	//mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
	mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, lpoints, num);
}

__kernel void ClCalculateIter1Mrqcof2Curve1(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local int num;  // __shared__
	__local double tmave[BLOCK_DIM];

	if (threadIdx.x == 0)
	{
		num = 0;
	}

	//mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
	mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, tmave, inrel, lpoints, num);
}

__kernel void ClCalculateIter1Mrqcof2Curve2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	//__global double* CUDA_Dytemp,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;

	//mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);

    __global double* alpha = (*CUDA_LCC).covar;
    __global double* beta = (*CUDA_LCC).da;

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
    if (inrel /*==1*/)
    {
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
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();

    if (threadIdx.x == 0)
    {
        (*CUDA_LCC).np1 += lpoints;
    }

    lnp2 = (*CUDA_LCC).np2;
    ltrial_chisq = (*CUDA_LCC).trial_chisq;

    if ((*CUDA_CC).ia[1]) //not relative
    {
        for (jp = 1; jp <= lpoints; jp++)
        {
            ymod = (*CUDA_LCC).ytemp[jp];

            int ixx = jp + matmpl * Lpoints1;
            for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
                (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

            lnp2++;
            sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);
            wght = (*CUDA_CC).Weight[lnp2];
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
    }
    else //relative ia[1]==0
    {
        for (jp = 1; jp <= lpoints; jp++)
        {
            ymod = (*CUDA_LCC).ytemp[jp];
            int ixx = jp + matmpl * Lpoints1;
            for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
            {
                (*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];  // jp[1] dytemp[315] 0.0 - ?!?  must be -1051420.6747227
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

            lnp2++;
            sig2i = 1 / ((*CUDA_CC).Sig[lnp2] * (*CUDA_CC).Sig[lnp2]);
            wght = (*CUDA_CC).Weight[lnp2];
            dy = (*CUDA_CC).Brightness[lnp2] - ymod;
            j = 0;
            //
            double sig2iwght = sig2i * wght;
            //l==1
            //
            for (l = 2; l <= (*CUDA_CC).lastone; l++)
            {

                j++;
                wt = (*CUDA_LCC).dyda[l] * sig2iwght; // jp[1]  dyda[2] == 0    - ?!? must be -1051420.6747227   *) See dytemp[]
                tmph = l / BLOCK_DIM;
                if (l % BLOCK_DIM) tmph++;
                tmpl = threadIdx.x * tmph;
                tmph = tmpl + tmph;
                if (tmph > l) tmph = l;
                tmpl++;
                //m==1
                if (tmpl == 1) tmpl++;
                //
                for (m = tmpl; m <= tmph; m++)
                {
                    //					  k++;
                    alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];
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

                    tmpl = latmpl;
                    //m==1
                    if (tmpl == 1) tmpl++;
                    //
                    for (m = tmpl; m <= latmph; m++)
                    {
                        //k++;
                        alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];
                    } /* m */
                    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
                    if (threadIdx.x == 0)
                    {
                        k = (*CUDA_CC).lastone - 1;
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
    }
    //     } always ==0 // Lastcall != 1

    if (threadIdx.x == 0)
    {
        (*CUDA_LCC).np2 = lnp2;
        (*CUDA_LCC).trial_chisq = ltrial_chisq;
    }
}

__kernel void ClCalculateIter1Mrqcof2Curve1Last(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	//double* dytemp = &CUDA_Dytemp[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local double res[BLOCK_DIM];

	//mrqcof_curve1_last(CUDA_LCC, CUDA_CC, dytemp, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, res, inrel, lpoints);
	mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, res, inrel, lpoints);
}

__kernel void ClCalculateIter1Mrqcof2End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar);

	//if (blockIdx.x == 0)
	//	printf("[%3d] Chisq: %10.7f\n", threadIdx.x, (*CUDA_LCC).Chisq);
}

__kernel void ClCalculateIter1Mrqmin2End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqmin2End\n");

	//mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
	mrqmin_2_end(CUDA_LCC, CUDA_CC);

	(*CUDA_LCC).Niter++;

	//if (blockIdx.x == 0)
	//	printf("[%3d] Niter: %d\n", threadIdx.x, (*CUDA_LCC).Niter);
	//printf("|");
}

__kernel void ClCalculateIter2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int i, j;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	//if (blockIdx.x == 0)
	//	printf("[%3d] isNiter: %d\n", threadIdx.x, (*CUDA_LCC).isNiter);

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
		{
			if (threadIdx.x == 0)
			{
				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			}

			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

			int brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
			if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
			int brtmpl = threadIdx.x * brtmph;
			brtmph = brtmpl + brtmph;
			if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
			brtmpl++;

			curv(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, brtmpl, brtmph);

			if (threadIdx.x == 0)
			{
				for (i = 1; i <= 3; i++)
				{
					(*CUDA_LCC).chck[i] = 0;


					for (j = 1; j <= (*CUDA_CC).Numfac; j++)
					{
						double qq;
						qq = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * (*CUDA_CC).Nor[j][i - 1];

						//if (blockIdx.x == 0)
						//	printf("[%d] [%d][%3d] qq: %10.7f, chck[%d]: %10.7f, Area[%3d]: %10.7f, Nor[%3d][%d]: %10.7f\n",
						//		blockIdx.x, i, j, qq, i, (*CUDA_LCC).chck[i], j, (*CUDA_LCC).Area[j], j, i - 1, (*CUDA_CC).Nor[j][i - 1]);

						(*CUDA_LCC).chck[i] = qq;
					}

					//if (blockIdx.x == 0)
					//	printf("[%d] chck[%d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).chck[i]);
				}

				//printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3]);

				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0)) * pow((*CUDA_CC).conw_r, 2.0);
				//(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2] + (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * ((*CUDA_CC).conw_r * (*CUDA_CC).conw_r);
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST

		if (threadIdx.x == 0)
		{
			//if (blockIdx.x == 0)
			//	printf("ndata - 3: %3d\n", (*CUDA_CC).ndata - 3);

			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / ((*CUDA_CC).ndata - 3));

			//if (blockIdx.x == 233)
			//{
			//	double dev_best = (*CUDA_LCC).dev_new * (*CUDA_LCC).dev_new * ((*CUDA_CC).ndata - 3);
			//	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
			//		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LCC).dev_new, dev_best);
			//}

			// NOTE: only if this step is better than the previous, 1e-10 is for numeric errors
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST
	}
}

__kernel void ClCalculateFinishPole(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global struct freq_result* CUDA_FR)
{
	int i;
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	//const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	double totarea = 0;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++)
	{
		totarea = totarea + (*CUDA_LCC).Area[i];
	}

	//if(blockIdx.x == 2)
	//	printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f, conw_r: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3], (*CUDA_CC).conw_r);

	//if (blockIdx.x == 2)
	//	printf("rchisq: %10.7f, Chisq: %10.7f \n", (*CUDA_LCC).rchisq, (*CUDA_LCC).Chisq);

	//const double sum = pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0);
	const double sum = ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1]) + ((*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]) + ((*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]);
	//printf("[FinishPole] [%d] sum: %10.7f\n", blockIdx.x, sum);

	const double dark = sqrt(sum);

	//if (blockIdx.x == 232 || blockIdx.x == 233)
	//	printf("[%d] sum: %12.8f, dark: %12.8f, totarea: %12.8f, dark_best: %12.8f\n", blockIdx.x, sum, dark, totarea, dark / totarea * 100);

	/* period solution */
	const double period = 2 * PI / (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3];

	/* pole solution */
	const double la_tmp = RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];

	//if (la_tmp < 0.0)
	//	printf("[CalculateFinishPole] la_best: %4.0f\n", la_tmp);

	const double be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];

	//if (blockIdx.x == 2)
		//printf("[%d] dev_new: %10.7f, dev_best: %10.7f\n", blockIdx.x, (*CUDA_LCC).dev_new, (*CUDA_LFR).dev_best);

	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
	{
		(*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
		(*CUDA_LFR).dev_best_x2 = (*CUDA_LCC).rchisq;
		(*CUDA_LFR).per_best = period;
		(*CUDA_LFR).dark_best = dark / totarea * 100;
		(*CUDA_LFR).la_best = la_tmp < 0 ? la_tmp + 360.0 : la_tmp;
		(*CUDA_LFR).be_best = be_tmp;

		//printf("[%d] dev_best: %12.8f\n", blockIdx.x, (*CUDA_LFR).dev_best);

		//if (blockIdx.x == 232)
		//{
		//	double dev_best = (*CUDA_LFR).dev_best * (*CUDA_LFR).dev_best * ((*CUDA_CC).ndata - 3);
		//	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
		//		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LFR).dev_best, dev_best);
		//}
	}

	if (isnan((*CUDA_LFR).dark_best) == 1)
	{
		(*CUDA_LFR).dark_best = 1.0;
	}

	//if (blockIdx.x == 2)
	//	printf("dark_best: %10.7f \n", (*CUDA_LFR).dark_best);

	//debug
	/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}
