
/* main-triangle elements per work-item: (DYT_STRIDE-1) rows at most */
#define C2_EPT (((DYT_STRIDE - 1) * DYT_STRIDE / 2 + BLOCK_DIM - 1) / BLOCK_DIM)

void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* alpha,
	__global double* beta,
	__local double (*dydaT)[DYT_STRIDE],
	__local double* s2wS,
	__local double* dwsS,
	__local double* dyS,
	__local double* coefS,		/* 2 * CURVE2_K: per-point coef, coef1 */
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

	/* The relative-lightcurve renormalization (ytemp *= coef, dytemp column 1
	   zeroed, dytemp[l] = coef * (dytemp[l] - coef1 * dave[l]) for l >= 2) used
	   to be a separate in-place pass over the whole curve in global memory
	   before the tiles re-read it. It is now applied while each tile is staged
	   into local memory - same expressions, same values - which saves a full
	   read + write of dytemp per curve (port of the HIP tree's fused I1
	   renormalization). dytemp/ytemp are per-curve scratch: nothing reads the
	   renormalized global copies afterwards. */
	const int lnp1b = (*CUDA_LCC).np1;	/* point jp uses Sig[lnp1b + jp] */
	const double ave = (*CUDA_LCC).ave;
	__local double* coef1S = coefS + CURVE2_K;

	/* everyone has read np1 before work-item 0 advances it */
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
	   applied while staging for a relative curve), 1-based parameter l. */
	int jp0, p, P;
	double wp[CURVE2_K];

	/* Main triangle (rows l = l0..lastone, columns m = l0..l) flattened over
	   the work-group: work-item t owns elements t, t + BLOCK_DIM, ... for the
	   whole curve and keeps their running alpha values in registers, adding
	   each tile's contribution exactly as the in-memory
	   alpha = alpha + acc update did (same values, same order), and writes
	   them back once at the end. Before, rows were swept one at a time with
	   only l of BLOCK_DIM work-items busy and every tile did a global
	   read-modify-write of the whole triangle. The two original index
	   variants, element for element:
	     absolute (ia[1] != 0): l0 = 1, alpha[l][m], beta[l]
	     relative (ia[1] == 0): l0 = 2, alpha[l-1][m-1], beta[l-1]
	   (frozen size scale). The ia-gated tail rows keep their code. */
	const int rel = !(*CUDA_CC).ia[1];
	const int l0 = rel ? 2 : 1;
	const int nrows = (*CUDA_CC).lastone - l0 + 1;
	const int ntri = nrows > 0 ? nrows * (nrows + 1) / 2 : 0;
	const int Mfit1 = (*CUDA_CC).Mfit1;
	int triLM[C2_EPT];		/* l * 64 + m of each owned element */
	double triA[C2_EPT];	/* its running alpha value */
	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		int e = threadIdx.x + t * BLOCK_DIM;
		triLM[t] = 0;
		triA[t] = 0;
		if (e < ntri)
		{
			/* row r of the triangle holds r + 1 elements */
			int r = (int)((sqrt(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
			while ((r + 1) * (r + 2) / 2 <= e) r++;
			while (r * (r + 1) / 2 > e) r--;
			int lr = l0 + r, mr = l0 + (e - r * (r + 1) / 2);
			triLM[t] = lr * 64 + mr;
			triA[t] = alpha[(lr - rel) * Mfit1 + (mr - rel)];
		}
	}
	const int browL = l0 + threadIdx.x;	/* beta row owned by this work-item */
	const int ownB = browL <= (*CUDA_CC).lastone;
	double betaR = ownB ? beta[browL - rel] : 0;

	for (jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
	{
		P = lpoints - jp0 + 1;
		if (P > CURVE2_K) P = CURVE2_K;

		/* per-point scalars; for a relative curve also the renormalization
		   factors, with the expressions of the former in-place pass */
		if (threadIdx.x < P)
		{
			jp = jp0 + threadIdx.x;
			ymod = ytempG[jp];
			if (inrel)
			{
				coef = ddiv((*CUDA_CC).Sig[lnp1b + jp] * lpoints, ave);
				coef1 = ddiv(ymod, ave);
				coefS[threadIdx.x] = coef;
				coef1S[threadIdx.x] = coef1;
				ymod = coef * ymod;
			}
			sig2i = ddiv(1.0, ((*CUDA_CC).Sig[lnp2 + jp] * (*CUDA_CC).Sig[lnp2 + jp]));
			wght = (*CUDA_CC).Weight[lnp2 + jp];
			dy = (*CUDA_CC).Brightness[lnp2 + jp] - ymod;
			double sig2iwght = sig2i * wght;
			s2wS[threadIdx.x] = sig2iwght;
			dwsS[threadIdx.x] = dy * sig2iwght;
			dyS[threadIdx.x] = dy;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* stage the tile (consecutive work-items copy consecutive addresses),
		   renormalizing on the way for a relative curve */
		for (m = threadIdx.x; m < P * DYT_STRIDE; m += BLOCK_DIM)
		{
			double v = dytempG[(jp0 - 1) * DYT_STRIDE + m];
			if (inrel)
			{
				p = m / DYT_STRIDE;
				l = m % DYT_STRIDE;
				if (l == 1)
					v = 0;	/* size-scale derivative is explicitly zero */
				else if (l >= 2 && l <= (*CUDA_CC).ma)
					v = coefS[p] * (v - coef1S[p] * (*CUDA_LCC).dave[l]);
			}
			((__local double*)&dydaT[0][0])[m] = v;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* main triangle: register-resident elements */
		#pragma unroll
		for (int t = 0; t < C2_EPT; t++)
		{
			if (threadIdx.x + t * BLOCK_DIM < ntri)
			{
				int lr = triLM[t] / 64, mr = triLM[t] % 64;
				double acc = 0;
				for (int pp = 0; pp < P; pp++)
				{
					double w = dydaT[pp][lr] * s2wS[pp];
					acc += w * dydaT[pp][mr];
				}
				triA[t] = triA[t] + acc;
			}
		}
		if (ownB)
		{
			double bacc = 0;
			for (int pp = 0; pp < P; pp++)
				bacc += dwsS[pp] * dydaT[pp][browL];
			betaR = betaR + bacc;
		}

		/* ia-gated tail rows l = lastone+1..lastma: unchanged */
		j = nrows > 0 ? nrows : 0;
		l = (*CUDA_CC).lastone + 1;
		if (l < l0) l = l0;
		for (; l <= (*CUDA_CC).lastma; l++)
		{
			if ((*CUDA_CC).ia[l])
			{
				j++;
				for (p = 0; p < P; p++)
					wp[p] = dydaT[p][l] * s2wS[p];

				tmpl = latmpl;
				if (rel && tmpl == 1) tmpl++;	//m==1
				for (m = tmpl; m <= latmph; m++)
				{
					double acc = 0;
					for (p = 0; p < P; p++)
						acc += wp[p] * dydaT[p][m];
					alpha[j * Mfit1 + m - rel] = alpha[j * Mfit1 + m - rel] + acc;
				} /* m */
				if (threadIdx.x == 0)
				{
					k = (*CUDA_CC).lastone - rel;
					for (m = (*CUDA_CC).lastone + 1; m <= l; m++)
					{
						if ((*CUDA_CC).ia[m])
						{
							k++;
							double acc = 0;
							for (p = 0; p < P; p++)
								acc += wp[p] * dydaT[p][m];
							alpha[j * Mfit1 + k] = alpha[j * Mfit1 + k] + acc;
						}
					} /* m */
					double bacc = 0;
					for (p = 0; p < P; p++)
						bacc += dwsS[p] * dydaT[p][l];
					beta[j] = beta[j] + bacc;
				}
			}
		} /* l */

		/* chi-square: same per-point terms in the same ascending order */
		for (p = 0; p < P; p++)
		{
			ltrial_chisq = ltrial_chisq + dyS[p] * dyS[p] * s2wS[p];
		}

		/* everyone must finish reading dydaT before the next tile overwrites it */
		barrier(CLK_LOCAL_MEM_FENCE);
	} /* jp0 */

	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		if (threadIdx.x + t * BLOCK_DIM < ntri)
		{
			int lr = triLM[t] / 64, mr = triLM[t] % 64;
			alpha[(lr - rel) * Mfit1 + (mr - rel)] = triA[t];
		}
	}
	if (ownB)
		beta[browL - rel] = betaR;

	lnp2 += lpoints;

	if (threadIdx.x == 0)
	{
		//printf("[%d] ltrial_chisq: %10.7f\n", blockIdx.x, ltrial_chisq);

		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}

