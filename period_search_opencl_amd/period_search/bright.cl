//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int lnp1,
	int Lpoints,
	int num,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global double* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global double* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global double* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global double* e_1G = scr + (*CUDA_CC).offE1;
	__global double* e_2G = scr + (*CUDA_CC).offE2;
	__global double* e_3G = scr + (*CUDA_CC).offE3;
	__global double* e0_1G = scr + (*CUDA_CC).offE01;
	__global double* e0_2G = scr + (*CUDA_CC).offE02;
	__global double* e0_3G = scr + (*CUDA_CC).offE03;
	__global double* deG = scr + (*CUDA_CC).offDe;
	__global double* de0G = scr + (*CUDA_CC).offDe0;
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

		alpha = acos(clamp(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3, -1.0, 1.0));


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-ddiv(alpha, cg[(*CUDA_CC).Ncoef0 + 2]));	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		jp_ScaleG[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);
		jp_dphp_1G[jp] = f;
		jp_dphp_2G[jp] = ddiv(cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha, cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		jp_dphp_3G[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, jp_ScaleG[jp], jp_dphp_1G[jp], jp_dphp_2G[jp], jp_dphp_3G[jp]);

		//  matrix start
		f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		sf = sincos(f, &cf);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3];
		e_1G[jp] = pom + tmat * ee_3;
		e0_1G[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, e_1G[jp], e0_1G[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3];
		e_2G[jp] = pom + tmat * ee_3;
		e0_2G[jp] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Blmat[3][3];
		e_3G[jp] = pom + tmat * ee_3;
		e0_3G[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3];
		deG[(jp) * 16 + (1) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3];
		deG[(jp) * 16 + (1) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (2)] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3];
		deG[(jp) * 16 + (1) * 4 + (3)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (1) * 4 + (3)] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3];
		deG[(jp) * 16 + (2) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3];
		deG[(jp) * 16 + (2) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (2)] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3];
		deG[(jp) * 16 + (2) * 4 + (3)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (2) * 4 + (3)] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[1][3][3];
		deG[(jp) * 16 + (3) * 4 + (1)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (3) * 4 + (1)] = pom0 + tmat * ee0_3;

		tmat = (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[2][3][3];
		deG[(jp) * 16 + (3) * 4 + (2)] = pom + tmat * ee_3;
		de0G[(jp) * 16 + (3) * 4 + (2)] = pom0 + tmat * ee0_3;


		deG[(jp) * 16 + (3) * 4 + (3)] = 0;
		de0G[(jp) * 16 + (3) * 4 + (3)] = 0;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  //__syncthreads();
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int jp,
	int Lpoints1,
	int Inrel,
	__global double* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global double* dytempG = scr + (*CUDA_CC).offDytemp;
	__global double* ytempG = scr + (*CUDA_CC).offYtemp;
	__global double* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global double* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global double* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global double* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global double* e_1G = scr + (*CUDA_CC).offE1;
	__global double* e_2G = scr + (*CUDA_CC).offE2;
	__global double* e_3G = scr + (*CUDA_CC).offE3;
	__global double* e0_1G = scr + (*CUDA_CC).offE01;
	__global double* e0_2G = scr + (*CUDA_CC).offE02;
	__global double* e0_3G = scr + (*CUDA_CC).offE03;
	__global double* deG = scr + (*CUDA_CC).offDe;
	__global double* de0G = scr + (*CUDA_CC).offDe0;
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	e_1 = e_1G[jp];
	e_2 = e_2G[jp];
	e_3 = e_3G[jp];
	e0_1 = e0_1G[jp];
	e0_2 = e0_2G[jp];
	e0_3 = e0_3G[jp];
	de[1][1] = deG[(jp) * 16 + (1) * 4 + (1)];
	de[1][2] = deG[(jp) * 16 + (1) * 4 + (2)];
	de[1][3] = deG[(jp) * 16 + (1) * 4 + (3)];
	de[2][1] = deG[(jp) * 16 + (2) * 4 + (1)];
	de[2][2] = deG[(jp) * 16 + (2) * 4 + (2)];
	de[2][3] = deG[(jp) * 16 + (2) * 4 + (3)];
	de[3][1] = deG[(jp) * 16 + (3) * 4 + (1)];
	de[3][2] = deG[(jp) * 16 + (3) * 4 + (2)];
	de[3][3] = deG[(jp) * 16 + (3) * 4 + (3)];
	de0[1][1] = de0G[(jp) * 16 + (1) * 4 + (1)];
	de0[1][2] = de0G[(jp) * 16 + (1) * 4 + (2)];
	de0[1][3] = de0G[(jp) * 16 + (1) * 4 + (3)];
	de0[2][1] = de0G[(jp) * 16 + (2) * 4 + (1)];
	de0[2][2] = de0G[(jp) * 16 + (2) * 4 + (2)];
	de0[2][3] = de0G[(jp) * 16 + (2) * 4 + (3)];
	de0[3][1] = de0G[(jp) * 16 + (3) * 4 + (1)];
	de0[3][2] = de0G[(jp) * 16 + (3) * 4 + (2)];
	de0[3][3] = de0G[(jp) * 16 + (3) * 4 + (3)];

	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	j = 1;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + ddiv(cls, dnom));
			ar = (*CUDA_LCC).Area[j];
			br += ar * s;

			incl[incl_count] = i;
			/* Darea[i] * s * Dg[i][k] == Darea[i] * s * g * Dsph[i][k]
			   == (Area[i] * s) * Dsph[i][k]: fold g into the weight and
			   gather from the one read-only, facet-major Dsph shared by
			   all work-groups instead of the per-context Dg matrix */
			dbr[incl_count] = ar * s;
			incl_count++;

			double lmu0_dnom = ddiv(lmu0, dnom);
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = ddiv(lmu, dnom);
			dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;


			sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
			sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
			sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
			sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * ddiv(lmu * lmu0, lmu + lmu0);
		}
	}

	Scale = jp_ScaleG[jp];
	i = (jp - 1) * DYT_STRIDE + (ncoef0 - 3 + 1);
	/* Ders. of brightness w.r.t. rotation parameters */
	dytempG[i] = Scale * tmp1;

	i++;
	dytempG[i] = Scale * tmp2;
	i++;
	dytempG[i] = Scale * tmp3;

	i++;
	/* Ders. of br. w.r.t. phase function params. */
	dytempG[i] = br * jp_dphp_1G[jp];
	i++;
	dytempG[i] = br * jp_dphp_2G[jp];
	i++;
	dytempG[i] = br * jp_dphp_3G[jp];

	/* Ders. of br. w.r.t. cl, cls */
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef - 1)] = Scale * tmp4 * cl;
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef)] = Scale * tmp5;

	/* Scaled brightness */
	ytempG[jp] = br * Scale;

	ncoef0 -= 3;
	int iStart;
	int d, d1, dr;

	iStart = Inrel + 1;
	d = (jp - 1) * DYT_STRIDE + iStart;

	d1 = d + 1;
	dr = 2;

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;
			double l_dbr = dbr[0];
			int l_incl = incl[0];
			tmp = l_dbr * (*CUDA_CC).Dsph[l_incl][i];
			int is_next_coef_valid = (i + 1) <= ncoef0;
			if (is_next_coef_valid)
			{
				tmp1 = l_dbr * (*CUDA_CC).Dsph[l_incl][i + 1];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];
				tmp += l_dbr * (*CUDA_CC).Dsph[l_incl][i];
				if (is_next_coef_valid)
				{
					tmp1 += l_dbr * (*CUDA_CC).Dsph[l_incl][i + 1];
				}
			}

			dytempG[d] = Scale * tmp;
			if (is_next_coef_valid)
			{
				dytempG[d1] = Scale * tmp1;
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d++)
			dytempG[d] = 0;
	}

	//return(0);
}
