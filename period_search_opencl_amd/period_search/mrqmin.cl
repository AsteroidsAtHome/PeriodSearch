//N.B. The foll. L-M routines are modified versions of Press et al.
//  converted from Mikko's fortran code

//  8.11.2006


int mrqmin_1_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local double* covL,
	__local double* daL,
	__local int* ipivL,
	__local double* shBig,
	__local int* shIrow,
	__local int* shIcol,
	__local double* pivBC,
	__local int* icolBC,
	__global double* alphaG)
{
	int j;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int ma = (*CUDA_CC).ma;

	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / BLOCK_DIM;
	if (ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	// <<< Iter1Mrqmin1EndPre1
	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}
	}
	// >>> Iter1Mrqmin1EndPre1 END

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	// The damped matrix is staged straight from alpha into local memory by
	// gauss_errc; covar is not touched at all (it is rezeroed by
	// ClCalculateIter1Mrqcof2Start before mrqcof2 accumulates into it).

	// <<< gauss_errc    ---- GAUS ERROR CODE ----
	int err_code = gauss_errc(CUDA_LCC, CUDA_CC, covL, daL, ipivL, shBig, shIrow, shIcol, pivBC, icolBC, alphaG);
	if (err_code)
	{
		return err_code;
	}
	//     __syncthreads(); inside gauss
	// <<< gaus_errc END

	// >>> Iter1Mrqmin1EndPost
	if (threadIdx.x == 0)
	{
		//		if (err_code != 0) return(err_code);  "bacha na sync threads" - Watch out for Sync Threads
		j = 0;
		for (int l = 1; l <= ma; l++)
			if ((*CUDA_CC).ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
			}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	// <<< Iter1Mrqmin1EndPost END

	return err_code;
}

void mrqmin_2_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* scr)
{
	__global double* alphaG = scr + (*CUDA_CC).offAlpha;
	__global double* covarG = scr + (*CUDA_CC).offCovar;

	int j, k, l;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		(*CUDA_LCC).Alamda = ddiv((*CUDA_LCC).Alamda, (*CUDA_CC).Alamda_incr);
		for (j = 1; j <= (*CUDA_CC).Mfit; j++)
		{
			for (k = 1; k <= (*CUDA_CC).Mfit; k++)
			{
				alphaG[j * (*CUDA_CC).Mfit1 + k] = covarG[j * (*CUDA_CC).Mfit1 + k];

				//if (blockIdx.x == 0)
				//	printf("alpha[%3d]: %10.7f\n", alphaG[j * (*CUDA_CC).Mfit1 + k]);
			}

			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		}
		for (l = 1; l <= (*CUDA_CC).ma; l++)
		{
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
		}
	}
	else
	{
		(*CUDA_LCC).Alamda = (*CUDA_CC).Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}


}
