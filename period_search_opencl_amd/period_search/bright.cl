//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int lnp1,
	int Lpoints,
	int num)
{
	__private double f, cf, sf, pom, pom0, alpha;
	__private double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = (*CUDA_CC).ee[lnp][0];		// position vectors
		ee0_1 = (*CUDA_CC).ee0[lnp][0];
		ee_2 = (*CUDA_CC).ee[lnp][1];
		ee0_2 = (*CUDA_CC).ee0[lnp][1];
		ee_3 = (*CUDA_CC).ee[lnp][2];
		ee0_3 = (*CUDA_CC).ee0[lnp][2];
		t = (*CUDA_CC).tim[lnp];

		//if (blockIdx.x == 0)
		//	printf("jp[%3d] lnp[%3d], %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",
		//		jp, lnp, ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3);

		//printf("tim[%3d]: %10.7f\n", lnp, t);
		//printf("lnp: %3d, ee[%d]: %.7f, ee0[%d]: %.7f\n", lnp, lnp * 3 + 0, (*CUDA_CC).ee[lnp][0], lnp, (*CUDA_CC).ee0[lnp][0]);

		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-alpha / cg[(*CUDA_CC).Ncoef0 + 2]);	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		(*CUDA_LCC).jp_Scale[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).jp_Scale[jp], (*CUDA_LCC).jp_dphp_1[jp], (*CUDA_LCC).jp_dphp_2[jp], (*CUDA_LCC).jp_dphp_3[jp]);

		//  matrix start
		f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		cf = cos(f);
		sf = sin(f);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, (*CUDA_LCC).e_1[jp], (*CUDA_LCC).e0_1[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_2[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_2[jp] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 1 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 1 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 1 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_3[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_3[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][1][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][1] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][1][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][2] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][1][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][3] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][2][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][1] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][2][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][2] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][2][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][3] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[1][1][1] + 0 * (*CUDA_LCC).Dblm[1][2][1] + 1 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][2] + 0 * (*CUDA_LCC).Dblm[1][2][2] + 1 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][3] + 0 * (*CUDA_LCC).Dblm[1][2][3] + 1 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][3][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][1] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[2][1][1] + 0 * (*CUDA_LCC).Dblm[2][2][1] + 1 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][2] + 0 * (*CUDA_LCC).Dblm[2][2][2] + 1 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][3] + 0 * (*CUDA_LCC).Dblm[2][2][3] + 1 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][3][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][2] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][3][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][3] = pom0 + tmat * ee0_3;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  //__syncthreads();
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int jp,
	int Lpoints1,
	int Inrel)
{
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//if (threadIdx.x == 0)
	//printf("[%3d] jp[%3d] dytemp[315]: %10.7f\n", blockIdx.x, jp, (*CUDA_LCC).dytemp[315]);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("cg[%d]: %10.7f, cg[%d]: %10.7f\n", ncoef - 1, cg[ncoef - 1], ncoef, cg[ncoef]);
	//	printf("cl: %10.7f, cls: %10.7f\n", cl, cls);
	//}

	/* matrix from neo */
	/* derivatives */

	e_1 = (*CUDA_LCC).e_1[jp];
	e_2 = (*CUDA_LCC).e_2[jp];
	e_3 = (*CUDA_LCC).e_3[jp];
	e0_1 = (*CUDA_LCC).e0_1[jp];
	e0_2 = (*CUDA_LCC).e0_2[jp];
	e0_3 = (*CUDA_LCC).e0_3[jp];
	de[1][1] = (*CUDA_LCC).de[jp][1][1];
	de[1][2] = (*CUDA_LCC).de[jp][1][2];
	de[1][3] = (*CUDA_LCC).de[jp][1][3];
	de[2][1] = (*CUDA_LCC).de[jp][2][1];
	de[2][2] = (*CUDA_LCC).de[jp][2][2];
	de[2][3] = (*CUDA_LCC).de[jp][2][3];
	de[3][1] = (*CUDA_LCC).de[jp][3][1];
	de[3][2] = (*CUDA_LCC).de[jp][3][2];
	de[3][3] = (*CUDA_LCC).de[jp][3][3];
	de0[1][1] = (*CUDA_LCC).de0[jp][1][1];
	de0[1][2] = (*CUDA_LCC).de0[jp][1][2];
	de0[1][3] = (*CUDA_LCC).de0[jp][1][3];
	de0[2][1] = (*CUDA_LCC).de0[jp][2][1];
	de0[2][2] = (*CUDA_LCC).de0[jp][2][2];
	de0[2][3] = (*CUDA_LCC).de0[jp][2][3];
	de0[3][1] = (*CUDA_LCC).de0[jp][3][1];
	de0[3][2] = (*CUDA_LCC).de0[jp][3][2];
	de0[3][3] = (*CUDA_LCC).de0[jp][3][3];

	//if (blockIdx.x == 0 && threadIdx.x == 0) //blockIdx.x == 0 &&
	//{
	//	printf("[%d] jp[%3d] %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n", blockIdx.x, jp, e_1, e_2, e_3, e0_1, e0_2, e0_3);

	//}
	//	printf("[%d] jp[%3d] de11: %10.7f, de12: %10.7f, de13: %10.7f\n", blockIdx.x, jp, de[1][1], de[1][2], de[1][3]);
		//printf("[%d] jp[%3d] e_1: %10.7f, e_2: %10.7f, e_3: %10.7f\n", blockIdx.x, jp, e_1, e_2, e_3);

	//	printf("[%3d] de0[3][3]: %10.7f\n", threadIdx.x, de[3][3]);
		//printf("[%3d] jp[%d] e_1: %10.7f,\te_2: %10.7f,\te_3: %10.7f\n", threadIdx.x, jp, e_1, e_2, e_3);

	/* Directions (and ders.) in the rotating system */

	//
	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	//   short int *incl=&(*CUDA_LCC).incl[threadIdx.x*MAX_N_FAC];
	//   double *dbr=&(*CUDA_LCC).dbr[threadIdx.x*MAX_N_FAC];
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];
	//int2 bfr;

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	//j = blockIdx.x * (CUDA_Numfac1)+1; // j = 1, 290, 579 etc.
	j = 1;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

		//if (blockIdx.x == 0 && threadIdx.x == 0 && i == 1) //blockIdx.x == 0 &&
		//	printf("[%d] jp[%3d] i[%3d] Nor[%d][0]: %10.7f, Nor[%d][1]: %10.7f, Nor[%d][2]: %10.7f\n",
		//		blockIdx.x, jp, i, (*CUDA_CC).Nor[i][0], i,  (*CUDA_CC).Nor[i][1], i, (*CUDA_CC).Nor[i][2]);
			//printf("[%d] jp[%3d] i[%3d] lmu: %10.7f, lmu0: %10.7f\n", blockIdx.x, jp, i, lmu, lmu0);

		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			//bfr=tex1Dfetch(texArea,j);
			//ar=__hiloint2double(bfr.y,bfr.x);

			ar = (*CUDA_LCC).Area[j];
			//if (blockIdx.x == 0 && threadIdx.x == 1) //blockIdx.x == 0 &&
			//	printf("[%d] s: %10.7f, Area[%3d]: %.7f (j: %5d)\n", blockIdx.x, s, i, ar, j);

			br += ar * s;
			//
			incl[incl_count] = i;
			dbr[incl_count] = (*CUDA_CC).Darea[i] * s;
			incl_count++;
			//
			//double dnom_lmu0 = (lmu0 / dnom); // *(lmu0 / dnom);
			//double dnom_lmu = (lmu / dnom); // *(lmu / dnom);
			//dsmu = cls * pow(dnom_lmu0, 2.0) + cl * lmu0;
			//dsmu0 = cls * pow(dnom_lmu, 2.0) + cl * lmu;

			//dsmu = cls * pow(lmu0 / dnom, 2.0) + cl * lmu0;
			//dsmu0 = cls * pow(lmu / dnom, 2.0) + cl * lmu;

			double lmu0_dnom = lmu0 / dnom;
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = lmu / dnom;
			dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;


			sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
			//if (threadIdx.x == 0 && i == 1)
			//	printf("[%d][%3d]jp[%3d] Nor[%d][0]: %10.7f, Nor[%d][1]: %10.7f, Nor[%d][2]: %10.7f, sum1: %10.7f\n",
			//		blockIdx.x, threadIdx.x, jp, i, CUDA_Nor[i][0], i, CUDA_Nor[i][1], i, CUDA_Nor[i][2], sum1);

			sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
			sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
			sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			//if (blockIdx.x == 9 && threadIdx.x == 10 && i <= 10)
			//	printf("[%d][%3d]jp[%3d] i[%4d] tmp1: %10.7f, tmp2: %10.7f, tmp3: %10.7f\n", blockIdx.x, threadIdx.x, jp, i, tmp1, tmp2, tmp3);
				//printf("[%d][%3d]jp[%3d] sum1: %10.7f, sum2: %10.7f, sum3: %10.7f\n", blockIdx.x, threadIdx.x, jp, sum1, sum2, sum3);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}

	Scale = (*CUDA_LCC).jp_Scale[jp];
	i = jp + (ncoef0 - 3 + 1) * Lpoints1;
	/* Ders. of brightness w.r.t. rotation parameters */
	(*CUDA_LCC).dytemp[i] = Scale * tmp1;

	//if (threadIdx.x == 0) //blockIdx.x == 0 &&
	//	printf("[%3d] jp[%3d] Scale: %10.7f, tmp1: %10.7f, dytemp[%5d]: %10.7f\n", blockIdx.x, jp, Scale, tmp1, i, (*CUDA_LCC).dytemp[i]);
	//	printf("[%3d] dytemp[%5d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).dytemp[i]);

	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp3;

	//if (blockIdx.x == 0)
	//	printf("[%d][%3d][%4d] Scale: %10.7f, tmp1: %10.7f, tmp2; %10.7f, tmp3: %10.7f\n", blockIdx.x, threadIdx.x, jp, Scale, tmp1, tmp2, tmp3);

	i += Lpoints1;
	/* Ders. of br. w.r.t. phase function params. */
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];

	/* Ders. of br. w.r.t. cl, cls */
	(*CUDA_LCC).dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
	(*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;

	/* Scaled brightness */
	(*CUDA_LCC).ytemp[jp] = br * Scale;

	//if (blockIdx.x == 0 && threadIdx.x == 0) //blockIdx.x == 0 &&
	//	printf("[%d][%d] br: %10.7f, Scale: %10.7f, ytemp[%d]: %10.6f\n", blockIdx.x, threadIdx.x, br, Scale, jp, (*CUDA_LCC).ytemp[jp]);

	ncoef0 -= 3;
	int m, m1, mr, iStart;
	int d, d1, dr;
	if (Inrel)
	{
		iStart = 2;
		//m = blockIdx.x * CUDA_Dg_block + 2 * (CUDA_Numfac1);
		m = 2 * (*CUDA_CC).Numfac1;
		d = jp + 2 * (Lpoints1);
	}
	else
	{
		iStart = 1;
		//m = blockIdx.x * CUDA_Dg_block + (CUDA_Numfac1);
		m = (*CUDA_CC).Numfac1;
		d = jp + (Lpoints1);
	}


	m1 = m + (*CUDA_CC).Numfac1;
	mr = 2 * (*CUDA_CC).Numfac1;
	d1 = d + Lpoints1;
	dr = 2 * Lpoints1;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("m: %d, m1: %d, Dg_block: %d, Numfac1: %d\n", m, m1, CUDA_Dg_block, CUDA_Numfac1);

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;

			double l_dbr = dbr[0];
			int l_incl = incl[0];

			//int2 xx;
			//xx=tex1Dfetch(texDg,m+l_incl);
			//tmp = l_dbr * __hiloint2double(xx.y,xx.x);
			tmp = l_dbr * (*CUDA_LCC).Dg[m + l_incl];

			//if (blockIdx.x == 0 && threadIdx.x == 0 && i < 10)
			//	printf("[%d] jp[%3d] i[%2d] l_dbr: %10.7f, Dg[%5d]: %10.7f\n", blockIdx.x, jp, i, l_dbr, m + l_incl, (*CUDA_LCC).Dg[m + l_incl]);

			if ((i + 1) <= ncoef0)
			{
				//xx=tex1Dfetch(texDg,m1+l_incl);
				//tmp1 = l_dbr * __hiloint2double(xx.y,xx.x);
				tmp1 = l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];

				//int2 xx;
				//xx=tex1Dfetch(texDg,m+l_incl);
				//tmp += l_dbr * __hiloint2double(xx.y,xx.x);
				tmp += l_dbr * (*CUDA_LCC).Dg[m + l_incl];
				if ((i + 1) <= ncoef0)
				{
					//xx=tex1Dfetch(texDg,m1+l_incl);
					//tmp1 += l_dbr * __hiloint2double(xx.y,xx.x);
					tmp1 += l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
				}
			}

			(*CUDA_LCC).dytemp[d] = Scale * tmp;

			//>>>>>>>>>
			// Check for these values at this point on first pass if any anomalies were suspected within results:
			//
			//[  4] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3356285, dytemp[  315]:  1.1364109
			//[  5] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3368231, dytemp[  315]:  1.1374274
			//[  2] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3322120, dytemp[  315]:  1.1335041
			//[  3] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3341985, dytemp[  315]:  1.1351942
			//[  9] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3412586, dytemp[  315]:  1.1412013
			//[  7] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3400172, dytemp[  315]:  1.1401451
			//[  8] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3400477, dytemp[  315]:  1.1401710
			//[  1] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3339482, dytemp[  315]:  1.1349812
			//[  6] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3382762, dytemp[  315]:  1.1386637
			//[  0] jp[  1] i[  2] Scale:  0.8508436, tmp:  1.3369873, dytemp[  315]:  1.1375671
			//
			//
			//if (threadIdx.x == 0 && jp == 1 && i == 2)
			//	printf("[%3d] jp[%3d] i[%3d] Scale: %10.7f, tmp: %10.7f, dytemp[%5d]: %10.7f\n", blockIdx.x, jp, i, Scale, tmp, d, (*CUDA_LCC).dytemp[d]);

			//if (threadIdx.x == 0 && i == 2)
			//	printf("[%3d] jp[%3d] i[%3d] Scale: %10.7f, tmp: %10.7f, dytemp[%5d]: %10.7f\n", blockIdx.x, jp, i, Scale, tmp, d, (*CUDA_LCC).dytemp[d]);

			//>>>>>>>>> dytemp [315]

			if ((i + 1) <= ncoef0)
			{
				(*CUDA_LCC).dytemp[d1] = Scale * tmp1;

				//if (threadIdx.x == 0 && jp == 1)
				//	printf("[%3d] jp[%3d] i[%3d] Scale: %10.7f, tmp1: %10.7f, dytemp[%5d]: %10.7f\n", blockIdx.x, jp, i, Scale, tmp1, d1, (*CUDA_LCC).dytemp[d1]);
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d += Lpoints1)
			(*CUDA_LCC).dytemp[d] = 0;
	}

	//return(0);
}
