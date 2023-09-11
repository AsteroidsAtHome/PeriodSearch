__kernel void Mrqcof2Curve1Bright(
	__global struct mfreq_context *CUDA_mCC,
    __global struct freq_context *CUDA_CC,
    int Inrel,
    int Lpoints)
{
	// double cl, cls, dnom, s, Scale;
	// double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	// int ncoef0, ncoef, i, j, incl_count = 0;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context *CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;
    if (!(*CUDA_LCC).isNiter) return;

    __private int Lpoints1 = Lpoints + 1;

    int jp;
    int brtmph, brtmpl;
    brtmph = Lpoints / BLOCK_DIM;
    if (Lpoints % BLOCK_DIM) brtmph++;
    brtmpl = threadIdx.x * brtmph;
    brtmph = brtmpl + brtmph;
    if (brtmph > Lpoints) brtmph = Lpoints;
    brtmpl++;

    for (jp = brtmpl; jp <= brtmph; jp++)
    {
        /*  ---  BRIGHT  ---  */
        // bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel);
        bright(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, jp, Lpoints1, Inrel);

		// ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
		// ncoef = (*CUDA_CC).ma;
		// cl = exp((*CUDA_LCC).atry[ncoef - 1]); /* Lambert */
		// cls = (*CUDA_LCC).atry[ncoef];       /* Lommel-Seeliger */

        // /* matrix from neo */
        // /* derivatives */
        // e_1 = (*CUDA_LCC).e_1[jp];
        // e_2 = (*CUDA_LCC).e_2[jp];
        // e_3 = (*CUDA_LCC).e_3[jp];
        // e0_1 = (*CUDA_LCC).e0_1[jp];
        // e0_2 = (*CUDA_LCC).e0_2[jp];
        // e0_3 = (*CUDA_LCC).e0_3[jp];
        // de[1][1] = (*CUDA_LCC).de[jp][1][1];
        // de[1][2] = (*CUDA_LCC).de[jp][1][2];
        // de[1][3] = (*CUDA_LCC).de[jp][1][3];
        // de[2][1] = (*CUDA_LCC).de[jp][2][1];
        // de[2][2] = (*CUDA_LCC).de[jp][2][2];
        // de[2][3] = (*CUDA_LCC).de[jp][2][3];
        // de[3][1] = (*CUDA_LCC).de[jp][3][1];
        // de[3][2] = (*CUDA_LCC).de[jp][3][2];
        // de[3][3] = (*CUDA_LCC).de[jp][3][3];
        // de0[1][1] = (*CUDA_LCC).de0[jp][1][1];
        // de0[1][2] = (*CUDA_LCC).de0[jp][1][2];
        // de0[1][3] = (*CUDA_LCC).de0[jp][1][3];
        // de0[2][1] = (*CUDA_LCC).de0[jp][2][1];
        // de0[2][2] = (*CUDA_LCC).de0[jp][2][2];
        // de0[2][3] = (*CUDA_LCC).de0[jp][2][3];
        // de0[3][1] = (*CUDA_LCC).de0[jp][3][1];
        // de0[3][2] = (*CUDA_LCC).de0[jp][3][2];
        // de0[3][3] = (*CUDA_LCC).de0[jp][3][3];

        // /*Integrated brightness (phase coeff. used later) */
        // double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
        // double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
        // short int incl[MAX_N_FAC];
        // double dbr[MAX_N_FAC];

        // br = 0;
        // tmp1 = 0;
        // tmp2 = 0;
        // tmp3 = 0;
        // tmp4 = 0;
        // tmp5 = 0;
        // j = 1;
        // for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
        // {
        //     lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
        //     lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

        //     if ((lmu > TINY) && (lmu0 > TINY))
        //     {
        //         dnom = lmu + lmu0;
        //         s = lmu * lmu0 * (cl + cls / dnom);
        //         ar = (*CUDA_LCC).Area[j];
        //         br += ar * s;

        //         incl[incl_count] = i;
        //         dbr[incl_count] = (*CUDA_CC).Darea[i] * s;
        //         incl_count++;

        //         double lmu0_dnom = lmu0 / dnom;
        //         dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
        //         double lmu_dnom = lmu / dnom;
        //         dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

        //         sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
        //         sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
        //         tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
        //         sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
        //         sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
        //         tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
        //         sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
        //         sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
        //         tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

        //         tmp4 += lmu * lmu0 * ar;
        //         tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
        //     }
        // }

        // Scale = (*CUDA_LCC).jp_Scale[jp];
        // i = jp + (ncoef0 - 3 + 1) * Lpoints1;
        // /* Ders. of brightness w.r.t. rotation parameters */
        // (*CUDA_LCC).dytemp[i] = Scale * tmp1;

        // i += Lpoints1;
        // (*CUDA_LCC).dytemp[i] = Scale * tmp2;
        // i += Lpoints1;
        // (*CUDA_LCC).dytemp[i] = Scale * tmp3;

        // i += Lpoints1;
        // /* Ders. of br. w.r.t. phase function params. */
        // (*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
        // i += Lpoints1;
        // (*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
        // i += Lpoints1;
        // (*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];

        // /* Ders. of br. w.r.t. cl, cls */
        // (*CUDA_LCC).dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
        // (*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;

        // /* Scaled brightness */
        // (*CUDA_LCC).ytemp[jp] = br * Scale;

        // ncoef0 -= 3;
        // int m, m1, mr, iStart;
        // int d, d1, dr;
        // if (Inrel)
        // {
        //     iStart = 2;
        //     //m = blockIdx.x * CUDA_Dg_block + 2 * (CUDA_Numfac1);
        //     m = 2 * (*CUDA_CC).Numfac1;
        //     d = jp + 2 * (Lpoints1);
        // }
        // else
        // {
        //     iStart = 1;
        //     //m = blockIdx.x * CUDA_Dg_block + (CUDA_Numfac1);
        //     m = (*CUDA_CC).Numfac1;
        //     d = jp + (Lpoints1);
        // }

        // m1 = m + (*CUDA_CC).Numfac1;
        // mr = 2 * (*CUDA_CC).Numfac1;
        // d1 = d + Lpoints1;
        // dr = 2 * Lpoints1;

        // /* Derivatives of brightness w.r.t. g-coeffs */
        // if (incl_count)
        // {
        //     for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
        //     {
        //         double tmp = 0, tmp1 = 0;
        //         double l_dbr = dbr[0];
        //         int l_incl = incl[0];
        //         tmp = l_dbr * (*CUDA_LCC).Dg[m + l_incl];
        //         if ((i + 1) <= ncoef0)
        //         {
        //             tmp1 = l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
        //         }

        //         for (j = 1; j < incl_count; j++)
        //         {
        //             double l_dbr = dbr[j];
        //             int l_incl = incl[j];
        //             tmp += l_dbr * (*CUDA_LCC).Dg[m + l_incl];
        //             if ((i + 1) <= ncoef0)
        //             {
        //                 tmp1 += l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
        //             }
        //         }

        //         (*CUDA_LCC).dytemp[d] = Scale * tmp;
        //         if ((i + 1) <= ncoef0)
        //         {
        //             (*CUDA_LCC).dytemp[d1] = Scale * tmp1;
        //         }
        //     }
        // }
        // else
        // {
        //     for (i = 1; i <= ncoef0; i++, d += Lpoints1)
        //         (*CUDA_LCC).dytemp[d] = 0;
        // }
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
}
