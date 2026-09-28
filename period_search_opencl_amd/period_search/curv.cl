 //Curvature function (and hence facet area) from Laplace series

 //  8.11.2006


void curv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	int brtmpl,
	int brtmph)
{
	int n;
	double fsum, g;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//        brtmpl:  1, 4, 7... 382
	//		  brtmph:  3, 6, 9... 288
	int q = 0;
	for (int i = brtmpl; i <= brtmph; i++, q++)
	{
		//if (blockIdx.x == 0)
		//	printf("i: %d\n", i);

		g = 0;
		n = 0;
		for (int m = 0; m <= (*CUDA_CC).Mmax; m++) // Mmax = 6
		{
			for (int l = m; l <= (*CUDA_CC).Lmax; l++)  // Lmax = 6
			{
				n++;
				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

				fsum = cg[n] * (*CUDA_CC).Fc[i][m];
				if (m != 0)
				{
					n++;
					//if (blockIdx.x == 0 && threadIdx.x == 0)
					//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

					fsum = fsum + cg[n] * (*CUDA_CC).Fs[i][m];
				}

				g = g + (*CUDA_CC).Pleg[i][l][m] * fsum;
			}
		}

		g = exp(g);
		(*CUDA_LCC).Area[i] = (*CUDA_CC).Darea[i] * g;

		//if (blockIdx.x == 0)
		//	printf("[%3d - %3d] i: %3d\n", q, threadIdx.x, i);

		//if (blockIdx.x == 0)
		//	printf("Area[%d]: %.7f\n", i, Area[i]);

		/* Dg is no longer materialized: Dg[i][k] == g * Dsph[i][k] folds into the
		   facet weights through Area (= Darea * g) - see bright.cl and conv.cl */
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();
}
