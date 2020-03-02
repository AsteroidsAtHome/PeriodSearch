/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

//#include <CL/cl.hpp>
////#include <math.h>
////#include "globals_CUDA.h"
//#include "globals.h"
//#include "declarations_OpenCl.h"

//__device__ void matrix_neo(freq_context* CUDA_LCC, double cg[], int lnp1, int Lpoints)
__kernel void matrix_neo(freq_context* CUDA_LCC, cl_double cg[], int lnp1, int Lpoints)
{
	cl_double f, cf, sf, pom, pom0, alpha;
	cl_double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	int lnp, jp;

	int brtmph, brtmpl;
	brtmph = Lpoints / CUDA_BLOCK_DIM;
	if (Lpoints % CUDA_BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;


	lnp = lnp1 + brtmpl - 1;
	for (jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;
		ee_1 = CUDA_ee[lnp * 3 + 0];// position vectors
		ee0_1 = CUDA_ee0[lnp * 3 + 0];
		ee_2 = CUDA_ee[lnp * 3 + 1];
		ee0_2 = CUDA_ee0[lnp * 3 + 1];
		ee_3 = CUDA_ee[lnp * 3 + 2];
		ee0_3 = CUDA_ee0[lnp * 3 + 2];
		t = CUDA_tim[lnp];

		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);
		/* Exp-lin model (const.term=1.) */
		f = exp(-alpha / cg[CUDA_ncoef0 + 2]);//f is temp here
		(*CUDA_LCC).jp_Scale[jp] = 1 + cg[CUDA_ncoef0 + 1] * f + cg[CUDA_ncoef0 + 3] * alpha;
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[CUDA_ncoef0 + 1] * f * alpha / (cg[CUDA_ncoef0 + 2] * cg[CUDA_ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

		//  matrix start
		f = cg[CUDA_ncoef0] * t + CUDA_Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		cf = cos(f);
		sf = sin(f);
		/* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;

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
	__syncthreads();
}


//__device__ double bright(freq_context* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel)
__kernel void bright(freq_context* CUDA_LCC, cl_double cg[], int jp, int Lpoints1, int Inrel, cl_double out)
{
	int ncoef0, ncoef, i, j, incl_count = 0;
	cl_double cl, cls, dnom, s, Scale;
	cl_double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];

	ncoef0 = CUDA_ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = CUDA_ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

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

	/* Directions (and ders.) in the rotating system */

	//
	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	//   short int *incl=&(*CUDA_LCC).incl[threadIdx.x*MAX_N_FAC];
	//   double *dbr=&(*CUDA_LCC).dbr[threadIdx.x*MAX_N_FAC];
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];
	int2 bfr;

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;
	j = blockIdx.x * (CUDA_Numfac1)+1;
	for (i = 1; i <= CUDA_Numfac; i++, j++)
	{
		lmu = e_1 * CUDA_Nor[i][0] + e_2 * CUDA_Nor[i][1] + e_3 * CUDA_Nor[i][2];
		lmu0 = e0_1 * CUDA_Nor[i][0] + e0_2 * CUDA_Nor[i][1] + e0_3 * CUDA_Nor[i][2];
		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			bfr = tex1Dfetch(texArea, j);
			ar = __hiloint2double(bfr.y, bfr.x);
			br += ar * s;
			//
			incl[incl_count] = i;
			dbr[incl_count] = CUDA_Darea[i] * s;
			incl_count++;
			//
			dsmu = cls * pow(lmu0 / dnom, 2) + cl * lmu0;
			dsmu0 = cls * pow(lmu / dnom, 2) + cl * lmu;

			sum1 = CUDA_Nor[i][0] * de[1][1] + CUDA_Nor[i][1] * de[2][1] + CUDA_Nor[i][2] * de[3][1];
			sum10 = CUDA_Nor[i][0] * de0[1][1] + CUDA_Nor[i][1] * de0[2][1] + CUDA_Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = CUDA_Nor[i][0] * de[1][2] + CUDA_Nor[i][1] * de[2][2] + CUDA_Nor[i][2] * de[3][2];
			sum20 = CUDA_Nor[i][0] * de0[1][2] + CUDA_Nor[i][1] * de0[2][2] + CUDA_Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = CUDA_Nor[i][0] * de[1][3] + CUDA_Nor[i][1] * de[2][3] + CUDA_Nor[i][2] * de[3][3];
			sum30 = CUDA_Nor[i][0] * de0[1][3] + CUDA_Nor[i][1] * de0[2][3] + CUDA_Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}
	Scale = (*CUDA_LCC).jp_Scale[jp];
	i = jp + (ncoef0 - 3 + 1) * Lpoints1;
	/* Ders. of brightness w.r.t. rotation parameters */
	(*CUDA_LCC).dytemp[i] = Scale * tmp1;
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp3;
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

	ncoef0 -= 3;
	int m, m1, mr, iStart;
	int d, d1, dr;
	if (Inrel)
	{
		iStart = 2;
		m = blockIdx.x * CUDA_Dg_block + 2 * (CUDA_Numfac1);
		d = jp + 2 * (Lpoints1);
	}
	else
	{
		iStart = 1;
		m = blockIdx.x * CUDA_Dg_block + (CUDA_Numfac1);
		d = jp + (Lpoints1);
	}
	m1 = m + (CUDA_Numfac1);
	mr = 2 * CUDA_Numfac1;
	d1 = d + (Lpoints1);
	dr = 2 * Lpoints1;
	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;

			double l_dbr = dbr[0];
			int l_incl = incl[0];

			int2 xx;
			xx = tex1Dfetch(texDg, m + l_incl);
			tmp = l_dbr * __hiloint2double(xx.y, xx.x);
			if ((i + 1) <= ncoef0)
			{
				xx = tex1Dfetch(texDg, m1 + l_incl);
				tmp1 = l_dbr * __hiloint2double(xx.y, xx.x);
			}
			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];

				int2 xx;
				xx = tex1Dfetch(texDg, m + l_incl);
				tmp += l_dbr * __hiloint2double(xx.y, xx.x);
				if ((i + 1) <= ncoef0)
				{
					xx = tex1Dfetch(texDg, m1 + l_incl);
					tmp1 += l_dbr * __hiloint2double(xx.y, xx.x);
				}
			}

			(*CUDA_LCC).dytemp[d] = Scale * tmp;
			if ((i + 1) <= ncoef0)
			{
				(*CUDA_LCC).dytemp[d1] = Scale * tmp1;
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d += Lpoints1)
			(*CUDA_LCC).dytemp[d] = 0;
	}

	return(0);
}
