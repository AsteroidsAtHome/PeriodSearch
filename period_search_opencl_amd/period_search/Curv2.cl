
void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha,
	__global double* beta,
	__local double (*dydaT)[DYT_STRIDE],
	__local double* s2wS,
	__local double* dwsS,
	__local double* dyS,
	int inrel,
	int lpoints,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
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
			int ixx = (jp - 1) * DYT_STRIDE + 1;
			/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
			dytempG[ixx] = 0;

			//if (blockIdx.x == 0)
			//	printf("[%d][%d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, dytempG[ixx]);

			coef = ddiv((*CUDA_CC).Sig[lnp1] * lpoints, (*CUDA_LCC).ave);

			//if (threadIdx.x == 0)
			//	printf("[%d][%3d][%d] coef: %10.7f\n", blockIdx.x, threadIdx.x, jp, coef);

			double yytmp = ytempG[jp];
			coef1 = ddiv(yytmp, (*CUDA_LCC).ave);

			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//	printf("[Device | mrqcof_curve2_1] [%3d]  yytmp[%3d]: %10.7f, ave: %10.7f\n", threadIdx.x, jp, yytmp, (*CUDA_LCC).ave);

			ytempG[jp] = coef * yytmp;

			//if (blockIdx.x == 0)
			//	printf("[Device][%d][%3d] ytemp[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, jp, ytempG[jp]);

			ixx++;

			//if (threadIdx.x == 0)
			//	printf("[%3d] jp[%3d] dytemp[%3d]: %10.7f\n", blockIdx.x, jp, ixx, dytempG[ixx]);

			for (l = 2; l <= (*CUDA_CC).ma; l++, ixx++)
			{
				dytempG[ixx] = coef * (dytempG[ixx] - coef1 * (*CUDA_LCC).dave[l]);

				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//	printf("[Device | mrqcof_curve2_1] [%3d]  coef1: %10.7f, dave[%3d]: %10.7f, dytemp[%3d]: %10.7f\n",
				//		threadIdx.x, coef1, l, (*CUDA_LCC).dave[l], ixx, dytempG[ixx]);
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

	/* 2026 rewrite: the normal equations are accumulated once per
	   CURVE2_K-point tile (a rank-K update from a local-memory-staged dyda
	   tile) instead of once per data point. The old code swept the whole
	   triangular alpha matrix in global memory with a read-modify-write per
	   point, plus TWO work-group barriers per matrix row per point; those
	   barriers protected nothing (the staged derivatives are read-only during
	   the sweep and every alpha/beta slot has exactly one writer), so the
	   tile needs just two barriers total. Both original index variants -
	   absolute (ia[1]!=0) and relative (ia[1]==0, column shift m-1, frozen
	   first parameter, gated tail rows) - are reproduced element for element;
	   within a tile only the summation order over the K points changes.

	   dydaT[p][l] is point jp0+p's staged derivative row (renormalization
	   already applied by the in-place pass above), 1-based parameter l. */
	int jp0, p, P;
	double wp[CURVE2_K];

	for (jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
	{
		P = lpoints - jp0 + 1;
		if (P > CURVE2_K) P = CURVE2_K;

		/* stage the tile: consecutive work-items copy consecutive addresses */
		for (m = threadIdx.x; m < P * DYT_STRIDE; m += BLOCK_DIM)
		{
			((__local double*)&dydaT[0][0])[m] = dytempG[(jp0 - 1) * DYT_STRIDE + m];
		}

		/* per-point scalars (ymod comes from the renormalized ytemp) */
		if (threadIdx.x < P)
		{
			jp = jp0 + threadIdx.x;
			ymod = ytempG[jp];
			sig2i = ddiv(1.0, ((*CUDA_CC).Sig[lnp2 + jp] * (*CUDA_CC).Sig[lnp2 + jp]));
			wght = (*CUDA_CC).Weight[lnp2 + jp];
			dy = (*CUDA_CC).Brightness[lnp2 + jp] - ymod;
			double sig2iwght = sig2i * wght;
			s2wS[threadIdx.x] = sig2iwght;
			dwsS[threadIdx.x] = dy * sig2iwght;
			dyS[threadIdx.x] = dy;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if ((*CUDA_CC).ia[1]) //not relative
		{
			j = 0;
			for (l = 1; l <= (*CUDA_CC).lastone; l++)
			{
				j++;
				for (p = 0; p < P; p++)
					wp[p] = dydaT[p][l] * s2wS[p];

				//precalc thread boundaries (same per-row partition as before)
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				for (m = tmpl; m <= tmph; m++)
				{
					double acc = 0;
					for (p = 0; p < P; p++)
						acc += wp[p] * dydaT[p][m];
					alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + acc;
				} /* m */
				if (threadIdx.x == 0)
				{
					double bacc = 0;
					for (p = 0; p < P; p++)
						bacc += dwsS[p] * dydaT[p][l];
					beta[j] = beta[j] + bacc;
				}
			} /* l */
			for (; l <= (*CUDA_CC).lastma; l++)
			{
				if ((*CUDA_CC).ia[l])
				{
					j++;
					for (p = 0; p < P; p++)
						wp[p] = dydaT[p][l] * s2wS[p];

					for (m = latmpl; m <= latmph; m++)
					{
						double acc = 0;
						for (p = 0; p < P; p++)
							acc += wp[p] * dydaT[p][m];
						alpha[j * (*CUDA_CC).Mfit1 + m] = alpha[j * (*CUDA_CC).Mfit1 + m] + acc;
					} /* m */
					if (threadIdx.x == 0)
					{
						k = (*CUDA_CC).lastone;
						for (m = (*CUDA_CC).lastone + 1; m <= l; m++)
						{
							if ((*CUDA_CC).ia[m])
							{
								k++;
								double acc = 0;
								for (p = 0; p < P; p++)
									acc += wp[p] * dydaT[p][m];
								alpha[j * (*CUDA_CC).Mfit1 + k] = alpha[j * (*CUDA_CC).Mfit1 + k] + acc;
							}
						} /* m */
						double bacc = 0;
						for (p = 0; p < P; p++)
							bacc += dwsS[p] * dydaT[p][l];
						beta[j] = beta[j] + bacc;
					}
				}
			} /* l */
		}
		else //relative ia[1]==0
		{
			j = 0;
			for (l = 2; l <= (*CUDA_CC).lastone; l++)
			{
				j++;
				for (p = 0; p < P; p++)
					wp[p] = dydaT[p][l] * s2wS[p];

				//precalc thread boundaries
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				//m==1: the frozen size-scale parameter is skipped
				if (tmpl == 1) tmpl++;
				for (m = tmpl; m <= tmph; m++)
				{
					double acc = 0;
					for (p = 0; p < P; p++)
						acc += wp[p] * dydaT[p][m];
					alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + acc;
				} /* m */
				if (threadIdx.x == 0)
				{
					double bacc = 0;
					for (p = 0; p < P; p++)
						bacc += dwsS[p] * dydaT[p][l];
					beta[j] = beta[j] + bacc;
				}
			} /* l */
			for (; l <= (*CUDA_CC).lastma; l++)
			{
				if ((*CUDA_CC).ia[l])
				{
					j++;
					for (p = 0; p < P; p++)
						wp[p] = dydaT[p][l] * s2wS[p];

					tmpl = latmpl;
					//m==1
					if (tmpl == 1) tmpl++;
					for (m = tmpl; m <= latmph; m++)
					{
						double acc = 0;
						for (p = 0; p < P; p++)
							acc += wp[p] * dydaT[p][m];
						alpha[j * (*CUDA_CC).Mfit1 + m - 1] = alpha[j * (*CUDA_CC).Mfit1 + m - 1] + acc;
					} /* m */
					if (threadIdx.x == 0)
					{
						k = (*CUDA_CC).lastone - 1;
						for (m = (*CUDA_CC).lastone + 1; m <= l; m++)
						{
							if ((*CUDA_CC).ia[m])
							{
								k++;
								double acc = 0;
								for (p = 0; p < P; p++)
									acc += wp[p] * dydaT[p][m];
								alpha[j * (*CUDA_CC).Mfit1 + k] = alpha[j * (*CUDA_CC).Mfit1 + k] + acc;
							}
						} /* m */
						double bacc = 0;
						for (p = 0; p < P; p++)
							bacc += dwsS[p] * dydaT[p][l];
						beta[j] = beta[j] + bacc;
					}
				}
			} /* l */
		}

		/* chi-square: same per-point terms in the same ascending order */
		for (p = 0; p < P; p++)
		{
			ltrial_chisq = ltrial_chisq + dyS[p] * dyS[p] * s2wS[p];
		}

		/* everyone must finish reading dydaT before the next tile overwrites it */
		barrier(CLK_LOCAL_MEM_FENCE);
	} /* jp0 */

	lnp2 += lpoints;

	if (threadIdx.x == 0)
	{
		//printf("[%d] ltrial_chisq: %10.7f\n", blockIdx.x, ltrial_chisq);

		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}

