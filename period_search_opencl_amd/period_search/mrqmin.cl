 //N.B. The foll. L-M routines are modified versions of Press et al.
 //  converted from Mikko's fortran code

 //  8.11.2006


int mrqmin_1_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC)
//int mrqmin_1_end(struct mfreq_context* CUDA_LCC, struct freq_context* CUDA_CC, int* sh_icol, int* sh_irow, double* sh_big, int icol, double pivinv)
{
	//const int* ia = (*CUDA_CC).ia;
	//const int ma = (*CUDA_CC).ma;
	//const int mfit = (*CUDA_CC).Mfit;
	//const int mfit1 = (*CUDA_CC).Mfit1;

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

	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * (*CUDA_CC).Mfit1 + 1;
		for (int k = 1; k <= (*CUDA_CC).Mfit; k++, ixx++)
		{
			(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];

			//if(blockIdx.x == 0 && threadIdx.x == 0 && ixx == 56)
			//	printf("[%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).alpha[ixx]); // On second pass alpha[56] = 0.0000 instead of 80.8776359 ?!?
				//printf("[%d][%3d] covar[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
		}

		int qq = j * (*CUDA_CC).Mfit1 + j;
		(*CUDA_LCC).covar[qq] = (*CUDA_LCC).alpha[qq] * (1 + (*CUDA_LCC).Alamda);

		//if (blockIdx.x == 0)
		//	printf("[%3d] j[%3d] alpha[%3d]: %10.7f, 1 + Alamda: %10.7f, covar[%3d]: %10.7f\n",
		//		threadIdx.x, j, qq, (*CUDA_LCC).alpha[qq], 1 + (*CUDA_LCC).Alamda, qq, (*CUDA_LCC).covar[qq]);

		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];

		//if (blockIdx.x == 0)
		//	printf("[%d][%3d] da[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).da[j]); // da -> OK

		//if(threadIdx.x == 1)
		//	printf("[%d] covar[%3d]: %10.7f, alpha[%3d]: %10.7f, (1 + Alamda: %10.7f)\n",
		//		blockIdx.x, qq, (*CUDA_LCC).covar[qq], qq, (*CUDA_LCC).alpha[qq], 1 + (*CUDA_LCC).Alamda);
	}

	//if(threadIdx.x == 0)
	//	printf("[%d] covar[56]: %10.7f\n", blockIdx.x,  (*CUDA_LCC).covar[56]);
	//sh_icol[threadIdx.x] = threadIdx.x;

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	// ---- GAUS ERROR CODE ----
	int err_code = gauss_errc(CUDA_LCC, CUDA_CC);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("mrqmin_1_end >>> [%3d] ma[%d3] err_code: %3d\n", threadIdx.x, ma, err_code);

	if (err_code)
	{
		return err_code;
	}

	//err_code = gauss_errc(CUDA_LCC, CUDA_mfit, (*CUDA_LCC).da);

	//     __syncthreads(); inside gauss

	if (threadIdx.x == 0)
	{

		//		if (err_code != 0) return(err_code);  "bacha na sync threads" - Watch out for Sync Threads

		j = 0;
		for (int l = 1; l <= ma; l++)
			if ((*CUDA_CC).ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];

				//if (blockIdx.x == 0 && j == 50)
				//	printf("[mrqmin_1_end] [%3d] atry[%3d]: %10.7f, cg[%3d]: %10.7f, da[%3d]: %10.7f\n",
				//		threadIdx.x, j, (*CUDA_LCC).atry[j], j, (*CUDA_LCC).cg[l], j, (*CUDA_LCC).da[j]);
			}
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return err_code;
}

void mrqmin_2_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC) //, int* ia, int ma)
{
	int j, k, l;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		(*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / (*CUDA_CC).Alamda_incr;
		for (j = 1; j <= (*CUDA_CC).Mfit; j++)
		{
			for (k = 1; k <= (*CUDA_CC).Mfit; k++)
			{
				(*CUDA_LCC).alpha[j * (*CUDA_CC).Mfit1 + k] = (*CUDA_LCC).covar[j * (*CUDA_CC).Mfit1 + k];

				//if (blockIdx.x == 0)
				//	printf("alpha[%3d]: %10.7f\n", (*CUDA_LCC).alpha[j * (*CUDA_CC).Mfit1 + k]);
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
