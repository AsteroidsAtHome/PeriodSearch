
void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha,
	__global double* beta,
	int inrel,
	int lpoints)
{
	int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
	double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);


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
	}
	else //relative ia[1]==0
	{

		//if (threadIdx.x == 0)
		//	printf("[%d] lastone: %3d\n", blockIdx.x, (*CUDA_CC).lastone);

		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("Curve2_2b >>> [%3d][%3d] jp[%3d] ymod: %10.7f\n", blockIdx.x, threadIdx.x, jp, ymod);

			int ixx = jp + matmpl * Lpoints1;
			for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];  // jp[1] dytemp[315] 0.0 - ?!?  must be -1051420.6747227

				//if (blockIdx.x == 0 && threadIdx.x == 1 && jp == 1)
				//	printf("[%2d][%3d] dytemp[%d]: %10.7f\n", blockIdx.x, jp, ixx, (*CUDA_LCC).dytemp[ixx]);
			}
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

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("Curve2_2b >>> [%3d][%3d] jp[%3d] sig2i: %10.7f, wght: %10.7f, dy: %10.7f\n", blockIdx.x, threadIdx.x, jp, sig2i, wght, dy);  // dy - ?

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

				//if (blockIdx.x == 0 && threadIdx.x == 1 && jp == 1 && j == 1)
				//	printf("[%2d][%2d] jp[%3d] j[%3d] wt: %10.7f, dyda[%d]: %10.7f, sig2iwght: %10.7f\n",
				//		blockIdx.x, threadIdx.x, jp, j, wt, l, (*CUDA_LCC).dyda[l], sig2iwght);

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
					//if (blockIdx.x == 0)
					//	printf("[%3d] tmpl: %3d, tmph: %3d\n", threadIdx.x, tmpl, tmph);
					//if (blockIdx.x == 0 && threadIdx.x == 1)
					//	printf(".");
					//					  k++;
					alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + wt * (*CUDA_LCC).dyda[m];

					//int qq = j * (*CUDA_CC).Mfit1 + m - 1;											// After the "_" in  Mrqcof1Curve2 "wt" & "dyda[2]" has ZEROES - ?!?
					//if (blockIdx.x == 0 && threadIdx.x == 1 && l == 2) // j == 1 like l = 2
					//	printf("curv2_2b>>>> [%2d][%3d] l[%3d] jp[%3d] alpha[%4d]: %10.7f, wt: %10.7f, dyda[%3d]: %10.7f\n",
					//		blockIdx.x, threadIdx.x, l, jp, qq, (*CUDA_LCC).alpha[qq], wt, m, (*CUDA_LCC).dyda[m]);
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

	 // if (((*CUDA_LCC).Lastcall == 1) && (CUDA_Inrel[i] == 1)) always ==0
		//(*CUDA_LCC).Sclnw[i] = (*CUDA_LCC).Scale * CUDA_Lpoints[i] * CUDA_sig[np]/ave;

	if (threadIdx.x == 0)
	{
		//printf("[%d] ltrial_chisq: %10.7f\n", blockIdx.x, ltrial_chisq);

		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}

