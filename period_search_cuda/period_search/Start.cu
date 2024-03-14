
#include <cstdio>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#include "constants.h"
#include "globals_CUDA.h"
#include "declarations_CUDA.h"

#include "cudamemasm.h"

// vars

__device__ double4 SCBLmat[N_BLOCKS];
__device__ double dave[N_BLOCKS][MAX_N_PAR + 1];
__device__ double atry[N_BLOCKS][MAX_N_PAR + 1];
__device__ double cgg[N_BLOCKS][MAX_N_PAR + 1];

__device__ double chck[N_BLOCKS];
__device__ unsigned int    Flags[N_BLOCKS];

#define isInvalid 1U
#define isNiter   2U
#define isAlambda 4U

__device__ void __forceinline__ setFlag(unsigned int i, int idx)
{
  unsigned int *a = &Flags[idx]; 
  __stwb(a, __ldg(a) | i); 
}

__device__ void __forceinline__ resetFlag(unsigned int i, int idx)
{
  unsigned int *a = &Flags[idx]; 
  __stwb(a, __ldg(a) & ~i); 
}

__device__ void __forceinline__ clearFlag(int idx)
{
  __stwb(&Flags[idx], 0);
}


__device__ unsigned int __forceinline__ getFlags(int idx)
{
  return __ldg(&Flags[idx]);
}


__device__ bool __forceinline__ isAllTrue(unsigned int flags, int idx)
{
  return (__ldg(&Flags[idx]) & flags) == flags;
}

__device__ bool __forceinline__ isAnyTrue(unsigned int flags, int idx)
{
  return (__ldg(&Flags[idx]) & flags) != 0;
}


__device__ double Alamda[N_BLOCKS];
__device__ int    Niter[N_BLOCKS];
__device__ double iter_diffg[N_BLOCKS];
__device__ double rchisqg[N_BLOCKS]; // not needed
__device__ double dev_oldg[N_BLOCKS];
__device__ double dev_newg[N_BLOCKS];

__device__ double trial_chisqg[N_BLOCKS];
__device__ double aveg[N_BLOCKS];
__device__ double raveg[N_BLOCKS]; // 1/aveg
__device__ int    npg[3][N_BLOCKS];

__device__ double Ochisq[N_BLOCKS];
__device__ double Chisq[N_BLOCKS];
__device__ double Areag[N_BLOCKS][MAX_N_FAC + 1];

//LFR
__managed__ int isReported[N_BLOCKS];
__managed__ double dark_best[N_BLOCKS];
__managed__ double per_best[N_BLOCKS];
__managed__ double dev_best[N_BLOCKS];
__managed__ double la_best[N_BLOCKS];
__managed__ double be_best[N_BLOCKS];


#define CUDA_Nphpar 3

//global to all freq
__constant__ int CUDA_Ncoef, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ int CUDA_ma, CUDA_mfit, /*CUDA_mfit1,*/ CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__constant__ double CUDA_cg_first[MAX_N_PAR + 1];
__constant__ int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__constant__ double CUDA_iter_diff_max;
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax, CUDA_Mmax;
__constant__ double CUDA_lcl, CUDA_Alamda_start, CUDA_Alamda_incr, CUDA_Alamda_incrr;
__constant__ double CUDA_Phi_0;
__constant__ double CUDA_beta_pole[N_POLES + 1];
__constant__ double CUDA_lambda_pole[N_POLES + 1];

__device__ double CUDA_par[4];
__device__ int CUDA_ia[MAX_N_PAR + 1];
__device__ double CUDA_Nor[3][MAX_N_FAC + 1];
__device__ double CUDA_Fc[MAX_LM+1][MAX_N_FAC + 1];
__device__ double CUDA_Fs[MAX_LM+1][MAX_N_FAC + 1];
__device__ double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Pleg1[MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Darea[MAX_N_FAC + 1];
__device__ double CUDA_Dsph[MAX_N_PAR + 1][MAX_N_FAC + 1];

__device__ double alphag[N_BLOCKS][64*64]; // 50 something
__device__ double betag[N_BLOCKS][MAX_N_PAR + 1];

//__device__ double *CUDA_Area;
__device__ double *CUDA_Dg;
__device__ int CUDA_End;
__device__ int CUDA_Is_Precalc;

//global to one thread
__device__ freq_context *CUDA_CC;

// big global variables
__device__ double CUDA_tim[MAX_N_OBS + 1];
__device__ double CUDA_brightness[MAX_N_OBS+1];
__device__ double CUDA_sig[MAX_N_OBS+1];
__device__ double CUDA_sigr2[MAX_N_OBS+1]; // (1/CUDA_sig^2)
__device__ double CUDA_Weight[MAX_N_OBS+1];
__device__ double CUDA_ee[3][MAX_N_OBS + 1];
__device__ double CUDA_ee0[3][MAX_N_OBS+1];



#define UNRL 4

// MRQMIN
__device__ int __forceinline__ mrqmin_1_end(freq_context * __restrict__ CUDA_LCC, int ma, int mfit, /*int mfit1,*/ const int block)
{
  int bid = blockIdx();
  int n = threadIdx.x + 1;
  double * __restrict__ ap = atry[bid] + n; 
  double const * __restrict__ cgp = cgg[bid] + n; 
  
  if(isAnyTrue(isAlambda, bid)) //__ldg(&isAlamda[bid]))
    {
#pragma unroll 1
      while(n <= ma - CUDA_BLOCK_DIM)
	{
	  double d1 = cgp[0];
	  double d2 = cgp[CUDA_BLOCK_DIM];
	  ap[0] = d1;
	  ap[CUDA_BLOCK_DIM] = d2;
	  n += 2 * CUDA_BLOCK_DIM;
	  ap += 2 * CUDA_BLOCK_DIM;
	  cgp += 2 * CUDA_BLOCK_DIM;
	}
      if(n <= ma)
	{
	  ap[0] = cgp[0];
	}
     }
  
  double ccc = 1 + __ldg(&Alamda[bid]); 

  unsigned int mfit1 = mfit + 1;
  unsigned int ixx = mfit1 + threadIdx.x + 1;
  
  double * __restrict__ a = CUDA_LCC->covar + ixx;
  double const * __restrict__ b = alphag[bid] + ixx - 1; 
#pragma unroll 1
  while(ixx < mfit1 * mfit1 - (UNRL - 1) * CUDA_BLOCK_DIM)
    {
      unsigned int i;
      double t[UNRL];
      bool bb[UNRL];
      for(i = 0; i < UNRL; i++)
	{
	  t[i] = __ldca(&b[0]);
	  bb[i] = ((ixx + i*CUDA_BLOCK_DIM) % (mfit1 + 1) == 0);
	  b += CUDA_BLOCK_DIM;
	}
      for(i = 0; i < UNRL; i++)
	{
	  if(bb[i])
	    t[i] = ccc * t[i];
	}
      for(i = 0; i < UNRL; i++)
	{
	  a[0] = t[i];
	  a += CUDA_BLOCK_DIM;
	}
      ixx += UNRL * CUDA_BLOCK_DIM;
    }
#pragma unroll 3
  while(ixx < mfit1 * mfit1)
    {
      double t = __ldca(&b[0]);
      if(ixx % (mfit1 + 1) == 0)
	a[0] = ccc * t;
      else
	a[0] = t;
      
      a += CUDA_BLOCK_DIM;
      b += CUDA_BLOCK_DIM;
      ixx += CUDA_BLOCK_DIM;
    }

  int xx = threadIdx.x + 1;
  double const * __restrict__ bp;
  double * __restrict__ dap;
  bp  = betag[bid] + xx - 1;
  dap = CUDA_LCC->da + xx;
#pragma unroll 2
  while(xx <= mfit - CUDA_BLOCK_DIM)
    {
      double v1 = bp[0];
      double v2 = bp[CUDA_BLOCK_DIM];
      dap[0] = v1;
      dap[CUDA_BLOCK_DIM] = v2;
      bp  += 2 * CUDA_BLOCK_DIM;
      dap += 2 * CUDA_BLOCK_DIM;
      xx  += 2 * CUDA_BLOCK_DIM;
    }
  if(xx <= mfit)
    {
      dap[0] = bp[0];
      bp  += CUDA_BLOCK_DIM;
      dap += CUDA_BLOCK_DIM;
      xx  += CUDA_BLOCK_DIM;
    }
  
  __syncwarp();

  int err_code = gauss_errc(CUDA_LCC, ma);
  if(err_code)
    {
      return err_code;
    }

  n = threadIdx.x;
  double const * __restrict__ ddap = CUDA_LCC->da + n;
  int const * __restrict__ iap = CUDA_ia + n + 1;
  ap  = atry[bid] + n + 1; 
  cgp = cgg[bid] + n + 1; 
#pragma unroll 1
  while(n < ma - (CUDA_BLOCK_DIM))
    {
      double s1, s2;
      bool  b1, b2;
      s1 = cgp[0] + __ldca(&ddap[0]);
      b1 = __ldca(&iap[0]);
      s2 = cgp[CUDA_BLOCK_DIM] + __ldca(&ddap[CUDA_BLOCK_DIM]);
      b2 = __ldca(&iap[CUDA_BLOCK_DIM]);
      if(b1)
	ap[0] = s1;
      if(b2)
	ap[CUDA_BLOCK_DIM] = s2;
      n   += 2*CUDA_BLOCK_DIM;
      iap += 2*CUDA_BLOCK_DIM;
      ap  += 2*CUDA_BLOCK_DIM;
      cgp += 2*CUDA_BLOCK_DIM;
      ddap += 2*CUDA_BLOCK_DIM;
    }
  //#pragma unroll 2
  if(n < ma)
    {
      double s1 = cgp[0] + __ldca(&ddap[0]);
      if(__ldca(&iap[0]))
	ap[0] = s1;
    }
  //__syncthreads(); 
  //if(threadIdx.x == 0)
  //  printf("<%lf>, ", atry[bid][CUDA_ncoef0+2]);
  return err_code;
}


// clean pointers and []'s
// threadify loops
__device__ void __forceinline__ mrqmin_2_end(freq_context * __restrict__ CUDA_LCC, int ma, int bid)
{
  int j, k, l; //, bid = blockIdx();
  double chisq = __ldg(&Chisq[bid]);
  double ochisq = __ldg(&Ochisq[bid]);
  int mf = CUDA_mfit;

  if(chisq < ochisq)
    {
      double rai = CUDA_Alamda_incr;
      double const * __restrict__ dap = CUDA_LCC->da + 1 + threadIdx.x;
      double * __restrict__ dbp = betag[bid] + 1 + threadIdx.x - 1;
      j = threadIdx.x;
#pragma unroll 1
      while(j < mf - (CUDA_BLOCK_DIM))
	{
	  double v1 = dap[0];
	  double v2 = dap[CUDA_BLOCK_DIM];
	  dbp[0] = v1;
	  dbp[CUDA_BLOCK_DIM] = v2;
	  j += 2*CUDA_BLOCK_DIM;
	  dbp += 2 * CUDA_BLOCK_DIM;
	  dap += 2 * CUDA_BLOCK_DIM;
	}
      if(j < mf)
	dbp[0] = dap[0];

      rai = CUDA_Alamda_incrr; //__drcp_rn(rai); ///1.0/rai;
      int mf1 = mf + 1;
      double Alm = __ldg(&Alamda[bid]);

      double const * __restrict__ cvpo = CUDA_LCC->covar + mf1 + threadIdx.x + 1;

      Alm *= rai;
      double *apo = alphag[bid] + mf1 + threadIdx.x + 1  - 1;

      Alamda[bid] = Alm;

#pragma unroll 1
      for(j = 0; j < mf; j++)
	{
	  double const * __restrict__ cvp = cvpo;
	  double * __restrict__ ap = apo;
	  k = threadIdx.x;
#pragma unroll 1
	  while(k < mf - (CUDA_BLOCK_DIM))
	    {
	      double v1 = cvp[0];
	      double v2 = cvp[CUDA_BLOCK_DIM];
	      ap[0]  = v1;
	      ap[CUDA_BLOCK_DIM]  = v2;
	      k += 2*CUDA_BLOCK_DIM;
	      cvp += 2 * CUDA_BLOCK_DIM;
	      ap  += 2 * CUDA_BLOCK_DIM;
	    }

	  if(k < mf)
	      ap[0] = cvp[0];//[0]; //ldcs

	  cvpo += mf1;
	  apo += mf1;
	}

      double const * __restrict__ atp = atry[bid] + 1 + threadIdx.x; 
      double * __restrict__ cgp = cgg[bid] + 1 + threadIdx.x;
      l = threadIdx.x;
#pragma unroll 1
      while(l < ma - (CUDA_BLOCK_DIM))
	{
	  double v1 = atp[0];
	  double v2 = atp[CUDA_BLOCK_DIM];
	  cgp[0] = v1;
	  cgp[CUDA_BLOCK_DIM] = v2;
	  l += 2 * CUDA_BLOCK_DIM;
	  atp += 2 * CUDA_BLOCK_DIM;
	  cgp += 2 * CUDA_BLOCK_DIM;
	}
      
      if(l < ma)
	*cgp = atp[0];
    }
  else
    if(threadIdx.x == 0)
      {
	double a, c;
	a = CUDA_Alamda_incr * __ldg(&Alamda[bid]); 
	c = ochisq; //Ochisq[bid];
	Alamda[bid] = a; 
	Chisq[bid] = c; 
      }

  return;
}

//MRQMIN ENDS



// COF
__device__ void __forceinline__ blmatrix(double bet, double lam, int tid)
{
  double cb, sb, cl, sl;

  sincos(bet, &sb, &cb);
  sincos(lam, &sl, &cl);
  double4 d;
  d.x = -sb;
  d.y = cb;
  d.z = -sl;
  d.w = cl;
  /*
  SCBLmat[0][tid] = -sb;
  SCBLmat[1][tid] =  cb;
  SCBLmat[2][tid] = -sl;
  SCBLmat[3][tid] =  cl;
  */
  SCBLmat[tid] = d;
}


// CURV
__device__ void __forceinline__ curv(freq_context const * __restrict__ CUDA_LCC, double * __restrict__ cg, int bid)
{
  int i, m, n, l, k;
  double g;
  
  int mm = CUDA_Mmax, lm = CUDA_Lmax;
  i = threadIdx.x;
  int nf = CUDA_Numfac;
  int nf1 = nf + 1;
  double * __restrict__ CUDA_Fcp = CUDA_Fc[0] + i;
  double * __restrict__ CUDA_Fsp = CUDA_Fs[0] + i;
  double * __restrict__ CUDA_Dareap = CUDA_Darea + i;
  double * __restrict__ dgpo = CUDA_LCC->Dg + (nf1 + i + 1);  
  double * __restrict__ dsphpo = CUDA_Dsph[0] + i + (MAX_N_FAC + 1);
  cg += 1;

#pragma unroll 1
  while(i < nf)
    {
      g = 0;
      n = 0;
      double const * __restrict__ cgp = cg; //cgg, atry
      double const * __restrict__ fcp = CUDA_Fcp;
      double const * __restrict__ fsp = CUDA_Fsp;

      m = 0;
      double fcim = __ldca(&fcp[0]); 
      double fsim = __ldca(&fsp[0]); 
      double const * __restrict__ CUDA_Plegp = &CUDA_Pleg[0][0][i];
      if(lm == 6 && mm == 6)
	{
	  lm = 6; mm = 6;
#pragma unroll 7
	  for(l = m; l <= lm; l++)
	    {
	      n++;
	      double fsum = __ldca(cgp++) * fcim; 
	      g += CUDA_Plegp[0] * fsum; 
	      CUDA_Plegp += (MAX_N_FAC + 1);
	    }
	  fcp += MAX_N_FAC + 1;
	  fsp += MAX_N_FAC + 1;
	  
#pragma unroll 6
	  for(m = 1; m <= mm; m++)
	    { 
	      fcim = __ldca(&fcp[0]); 
	      fsim = __ldca(&fsp[0]); 
	      CUDA_Plegp = &CUDA_Pleg[m][m][i];
#pragma unroll 6
	      for(l = m; l <= lm; l++)
		{
		  n++;
		  double fsum = __ldca(cgp++) * fcim; 
		  n++;
		  fsum += __ldca(cgp++) * fsim; 
		  g += CUDA_Plegp[0] * fsum; 
		  CUDA_Plegp += (MAX_N_FAC + 1);
		}
	      fcp += MAX_N_FAC + 1;
	      fsp += MAX_N_FAC + 1;
	    }
	}
      else
	{
#pragma unroll 7
	  for(l = m; l <= lm; l++)
	    {
	      n++;
	      double fsum = __ldca(cgp++) * fcim; 
	      g += CUDA_Plegp[0] * fsum; 
	      CUDA_Plegp += (MAX_N_FAC + 1);
	    }
	  fcp += MAX_N_FAC + 1;
	  fsp += MAX_N_FAC + 1;

#pragma unroll 6
	  for(m = 1; m <= mm; m++)
	    { 
	      fcim = __ldca(&fcp[0]); 
	      fsim = __ldca(&fsp[0]); 
	      CUDA_Plegp = &CUDA_Pleg[m][m][i];
#pragma unroll 
	      for(l = m; l <= lm; l++)
		{
		  n++;
		  double fsum = __ldca(cgp++) * fcim; 
		  n++;
		  fsum += __ldca(cgp++) * fsim; 
		  g += CUDA_Plegp[0] * fsum; 
		  CUDA_Plegp += (MAX_N_FAC + 1);
		}
	      fcp += MAX_N_FAC + 1;
	      fsp += MAX_N_FAC + 1;
	    }
	}
      //if(threadIdx.x == 0 && blockIdx.x == 0)
      //  printf("%d %d %d, ", lm, mm, n);
      n = 49;
      double dd = __ldg(&CUDA_Dareap[0]);
      g = exp(g);
      dd *= g;
      
      double * __restrict__ dgp = dgpo; //double * dgpo = CUDA_LCC->Dg + (nf1 + i + 1);
      double const * __restrict__ dsphp = dsphpo; //double *dsphpo = CUDA_Dsph[0] + i + (MAX_N_FAC + 1);

      Areag[bid][i] = dd;
      k = 1;

      unsigned int nfu = (unsigned int)nf1;
      if(nfu == 289)
	{
	  nfu = 289;
#pragma unroll 
	  while(k <= n - (UNRL - 1))
	    {
	      double a[UNRL];

#pragma unroll 
	      for(int nn = 0; nn < UNRL; nn++)
		{
		  a[nn] = __ldca(dsphp) * g; //ldca
		  dsphp += (MAX_N_FAC + 1);
		}

#pragma unroll 
	      for(int nn = 0; nn < UNRL; nn++)
		{
		  *dgp = a[nn];
		  dgp += nfu; 
		}
	      k += UNRL;
	    }

#pragma unroll 3
	  while(k <= n)
	    {
	      *dgp = __ldca(dsphp) * g;
	      dsphp += (MAX_N_FAC + 1);
	      k++;
	      dgp += nfu;
	    }
	}
      else
	{
#pragma unroll 1
	  while(k <= n - (UNRL - 1))
	    {
	      double a[UNRL];
	      
#pragma unroll 
	      for(int nn = 0; nn < UNRL; nn++)
		{
		  a[nn] = __ldca(dsphp) * g; //ldca
		  dsphp += (MAX_N_FAC + 1);
		}
	      
#pragma unroll 
	      for(int nn = 0; nn < UNRL; nn++)
		{
		  __stwb(dgp, a[nn]);
		  dgp += nfu; 
		}
	      k += UNRL;
	    }
	  
#pragma unroll 3
	  while(k <= n)
	    {
	      __stwb(dgp, __ldca(dsphp) * g);
	      dsphp += (MAX_N_FAC + 1);
	      k++;
	      dgp += nfu;
	    }
	}
      i += CUDA_BLOCK_DIM;
      CUDA_Fcp += CUDA_BLOCK_DIM;
      CUDA_Fsp += CUDA_BLOCK_DIM;
      CUDA_Dareap += CUDA_BLOCK_DIM;
      dgpo   += CUDA_BLOCK_DIM;
      dsphpo += CUDA_BLOCK_DIM;
    }
  //__syncwarp();
}

// CURV end




__device__ void __forceinline__ mrqcof_start(freq_context * __restrict__ CUDA_LCC,
					     double * __restrict__ a,
					     double * __restrict__ alpha,
					     double * __restrict__ beta,
					     int bid)
{
  int j, k;

  /* N.B. curv and blmatrix called outside bright
     because output same for all points */
  int mf1 = CUDA_mfit + 1;
  curv(CUDA_LCC, a, bid);

#pragma unroll 4
  for(j = 1; j < mf1; j++)
    {
      alpha += mf1;
      k = threadIdx.x + 1;
      double *alphap = alpha + k;
#pragma unroll 1
      while(k <= j - 1)
	{ 
	  __stwb(alphap, 0.0);
	  __stwb(alphap + CUDA_BLOCK_DIM, 0.0);
	  k += 2 * CUDA_BLOCK_DIM;
	  alphap += 2 * CUDA_BLOCK_DIM;
	}
      if(k <= j)
	{ 
	  __stwb(alphap, 0.0);
	}
    }
  
  j = threadIdx.x + 1;
  double *betap = beta + j;
#pragma unroll 1
  while(j < mf1 - (CUDA_BLOCK_DIM))
    {
      __stwb(betap, 0.0);
      __stwb(betap + CUDA_BLOCK_DIM, 0.0);
      j += 2 * CUDA_BLOCK_DIM;
      betap += 2 * CUDA_BLOCK_DIM;
    }
  if(j < mf1)
    {
      __stwb(betap, 0.0);
      j += CUDA_BLOCK_DIM;
      betap += CUDA_BLOCK_DIM;
    }
  
  // __syncthreads(); //pro jistotu
}



__device__ double __forceinline__ mrqcof_end(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha)
{
  int j, k, mf = CUDA_mfit;
  int mf1 = mf + 1;
  int tid = threadIdx.x;
  double * __restrict__ app = alpha + mf1 + 2 + tid;
  double const * __restrict__ ap2 = alpha + (2 + tid) * mf1;
  long long int mf1add = sizeof(double) * mf1;
#pragma unroll 1
   for(j = 1 + tid; j < mf; j += blockDim.x)
     {
       double * __restrict__ ap = app;
       k = 1;
#pragma unroll 32
       while(k <= j - 1)
         {
	   __stwb(&ap[0], ap2[k++]);
	   ap = ap + mf1;
	   __stwb(&ap[0], ap2[k++]);
	   ap = ap + mf1;
	   /*
	   __stwb(ap, __ldca(&ap2[k]));
	   k++;
	   ap  = (double *)(((char *)ap) + mf1add);
	   __stwb(ap, __ldca(&ap2[k]));
	   k++;
	   ap  = (double *)(((char *)ap) + mf1add);
	   */
	 }
       if(k <= j)
         {
	   __stwb(ap, __ldca(&ap2[k]));
	 }
       app += blockDim.x;
       //ap2 += mf1;
       ap2  = (double *)(((char *)ap2) + mf1add * blockDim.x);
     }

   return 0; //trial_chisqg[bid];
}


// 47%
__device__ void __forceinline__ mrqcof_curve1(freq_context * __restrict__ CUDA_LCC,
					      double const * __restrict__ a,
					      int Inrel, int Lpoints, int bid)
{
  __shared__ double nc00s;
  __shared__ double nc01s;

  __shared__ double nc03s;
  __shared__ double nc02rs;
  __shared__ double phi0s;
  __shared__ double nc02r2s;
  __shared__ double scl, scls;

  double nc02r, phi0, nc02r2;
  double nc00, nc01, nc03;

  int Lpoints1 = Lpoints + 1;
  double lave = 0;

  int n = threadIdx.x + 1;
  if(Inrel == 1)
    {
      int blockidx = bid;
      int lnp1 = npg[0][bid];
      if(threadIdx.x == 0)
	{ //  a = cgg and atry
	  int nc = CUDA_ncoef0;
	  int ma = CUDA_ma;
	  double tmp = a[nc + 2];
	  //printf("%lf, ", tmp);
	  nc03s = a[nc + 3];
	  nc00s = a[nc + 0];
	  nc01s = a[nc + 1];
	  //if(cache[0][bid] == tmp)
	  //  {
	  //cacheHit++;
	  // nc02r = nc02rs = cache[1][bid];
	  //  }
	  //else
	  //  {
	  //cacheMiss++;
	  nc02r = nc02rs = __drcp_rn(tmp);
	  //    cache[0][bid] = tmp;
	  //cache[1][bid] = nc02r;
	  //}
	  phi0s = CUDA_Phi_0;
	  nc02r2s = nc02r * nc02r;
	  scl = exp(a[ma - 1]); /* Lambert */
	  scls = a[ma];       /* Lommel-Seeliger */
	}
      __syncwarp();
      double4 d = SCBLmat[blockidx];
      double Blmat02 = d.x; //__ldca(&SCBLmat[0][blockidx]);
      double Blmat10 = d.z; //__ldca(&SCBLmat[2][blockidx]);
      double Blmat11 = d.w; //__ldca(&SCBLmat[3][blockidx]);
      double Blmat22 = d.y; //__ldca(&SCBLmat[1][blockidx]);

#pragma unroll 1
      while(n <= Lpoints) 
	{
	  int jp = n - 1;
	  double f, cf, sf, alpha;
	  double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t; //, tmat1, tmat2, tmat3;

	  int lnp = lnp1 + jp;

	  ee_1  = CUDA_ee[0][lnp];// position vectors
	  ee0_1 = CUDA_ee0[0][lnp]; 
	  ee_2  = CUDA_ee[1][lnp];
	  ee0_2 = CUDA_ee0[1][lnp];
	  ee_3  = CUDA_ee[2][lnp];
	  ee0_3 = CUDA_ee0[2][lnp];
	  t = CUDA_tim[lnp];

	  alpha = acos(((ee_1 * ee0_1) + ee_2 * ee0_2) + ee_3 * ee0_3);
	  nc00 = nc00s;
	  phi0 = phi0s;
	  f = nc00 * t + phi0;

	  /* Exp-lin model (const.term=1.) */
	  nc02r = nc02rs;
	  double ff = exp2(-1.44269504088896 * (alpha * nc02r));

	  /* fmod may give little different results than Mikko's */
	  f = f - 2.0 * PI * round(f * (1.0 / (2.0 * PI))); //3:41.9

	  nc01 = nc01s;
	  nc03 = nc03s;
	  nc02r2 = nc02r2s;

	  double scale = 1.0 + nc01 * ff + nc03 * alpha;
	  double d2 =  nc01 * ff * alpha * nc02r2;

	  //  matrix start

	  __builtin_assume(f > (-2.0 * PI) && f < (2.0 * PI));
	  sincos(f, &sf, &cf);
	  double Blmat00 = Blmat11 * Blmat22;
	  double Blmat01 = Blmat22 * -Blmat10;
	  double msf = -sf;
	  double cbl00 = cf * Blmat00;
	  double sbl10 = sf * Blmat10;
	  double cbl10 = cf * Blmat10;
	  double sbl11 = sf * Blmat11;
	  double cbl11 = cf * Blmat11;
	  double cbl01 = cf * Blmat01;
	  double sbl00 = msf * Blmat00;
	  double sbl01 = msf * Blmat01;

	  double gde020 = Blmat00 * ee_1;
	  double gde120 = Blmat00 * ee0_1;

	  double tmat41 = -cbl01 - sbl11;
	  double tmat51 = -sbl01 - cbl11;
	  double tmat42 = cbl00 + sbl10;
	  double tmat52 = sbl00 + cbl10;

	  gde020 += Blmat01 * ee_2;
	  gde120 += Blmat01 * ee0_2;

	  double gde001 = tmat41 * ee_1;
	  double gde101 = tmat41 * ee0_1;
	  double gde011 = tmat51 * ee_1;
	  double gde111 = tmat51 * ee0_1;

	  gde001 += tmat42 * ee_2;
	  gde101 += tmat42 * ee0_2;
	  gde011 += tmat52 * ee_2;
	  gde111 += tmat52 * ee0_2;

	  gde020 += Blmat02 * ee_3;
	  gde120 += Blmat02 * ee0_3;

	  double tmat01 = cbl00 + sbl10;
	  double tmat11 = sbl00 + cbl10;
	  double tmat02 = cbl01 + sbl11;
	  double tmat12 = sbl01 + cbl11;
	  double tmat03 = cf  * Blmat02;
	  double tmat13 = msf * Blmat02;

	  double ge00 = tmat01 * ee_1;
	  double ge10 = tmat01 * ee0_1;
	  double ge01 = tmat11 * ee_1;
	  double ge11 = tmat11 * ee0_1;

	  ge00 += tmat02 * ee_2;
	  ge10 += tmat02 * ee0_2;
	  ge01 += tmat12 * ee_2;
	  ge11 += tmat12 * ee0_2;

	  ge00 += tmat03 * ee_3;
	  ge10 += tmat03 * ee0_3;
	  ge01 += tmat13 * ee_3;
	  ge11 += tmat13 * ee0_3;

	  double Blmat20 = Blmat11 * -Blmat02;
	  double Blmat21 = Blmat02 * Blmat10;
	  double gde002 = t * ge01;
	  double gde102 = t * ge11;
	  double gde012 = -t * ge00;
	  double gde112 = -t * ge10;

	  double ge02 = Blmat20 * ee_1;
	  double ge12 = Blmat20 * ee0_1;
	  double gde021 = -Blmat21 * ee_1;
	  double gde121 = -Blmat21 * ee0_1;

	  double tmat31 = sf * Blmat20; 
	  double tmat32 = sf * Blmat21; 
	  double tmat33 = sf * Blmat22; 
	  double tmat21 = cf * -Blmat20; 
	  double tmat22 = cf * -Blmat21;  
	  double tmat23 = cf * -Blmat22;

	  ge02 += Blmat21 * ee_2;
	  ge12 += Blmat21 * ee0_2;
      gde021 += Blmat20 * ee_2;
	  gde121 += Blmat20 * ee0_2;

	  double gde000 = tmat21 * ee_1;
	  double gde100 = tmat21 * ee0_1;
	  double gde010 = tmat31 * ee_1;
	  double gde110 = tmat31 * ee0_1;

	  ge02 += Blmat22 * ee_3;
	  ge12 += Blmat22 * ee0_3;

	  gde000 += tmat22 * ee_2;
	  gde100 += tmat22 * ee0_2;
	  gde010 += tmat32 * ee_2;
	  gde110 += tmat32 * ee0_2;

	  gde000 += tmat23 * ee_3;
	  gde100 += tmat23 * ee0_3;
	  gde010 += tmat33 * ee_3;
	  gde110 += tmat33 * ee0_3;

	  int incl_count = 0;
	  int i, j; //, blockidx = blockIdx();
	  //double dnom, s; //, Scale;
	  //int ma = CUDA_ma;
	  //cl = exp(a[ma - 1]); /* Lambert */
	  //cls = a[ma];       /* Lommel-Seeliger */


	  /*Integrated brightness (phase coeff. used later) */
	  double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	  double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;

	  short int incl[MAX_N_FAC];
	  double dbr[MAX_N_FAC];

	  //int2 bfr;
	  int nf = CUDA_Numfac;
	  int nf1 = nf  + 1;

	  int bid = blockidx;
	  br   = 0;
	  tmp1 = 0;
	  tmp2 = 0;
	  tmp3 = 0;
	  tmp4 = 0;
	  tmp5 = 0;
	  j = bid * nf1 + 1;
	  double const * __restrict__ norp0;
	  double const * __restrict__ norp1;
	  double const * __restrict__ norp2;
	  double const * __restrict__ areap;
	  double const * __restrict__ dareap; 
	  norp0 = CUDA_Nor[0];
	  norp1 = CUDA_Nor[1];
	  norp2 = CUDA_Nor[2];
	  //areap = CUDA_Area;
	  areap = &(Areag[bid][0]);
	  dareap = CUDA_Darea;
	  double cl = scl, cls = scls;
#pragma unroll 1
	  for(i = 0; i < nf && i < MAX_N_FAC; i++, j++)
	    {
	      double n0 = norp0[i], n1 = norp1[i], n2 = norp2[i];
	      lmu  = ge00 * n0 + ge01 * n1 + ge02 * n2;
	      lmu0 = ge10 * n0 + ge11 * n1 + ge12 * n2;
	      //if((lmu > TINY) && (lmu0 > TINY))
	      //{	
	      if((lmu <= TINY) || (lmu0 <= TINY))
		{
		  continue;
		}
	      double dnom = lmu + lmu0;
	      ar = __ldca(&areap[i]);

	      double dnom_1 = __drcp_rn(dnom); 

	      double s = lmu * lmu0 * (cl + cls * dnom_1);
	      double lmu0_dnom = lmu0 * dnom_1;

	      br += ar * s;
	      //
	      dbr[incl_count] = __ldca(&dareap[i]) * s;
	      incl[incl_count] = i + 1;
	      incl_count++;

	      double lmu_dnom = lmu * dnom_1;
	      dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
	      dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

	      sum1  = n0 * gde000 + n1 * gde010 + n2 * gde020;
	      sum10 = n0 * gde100 + n1 * gde110 + n2 * gde120;
	      sum2  = n0 * gde001 + n1 * gde011 + n2 * gde021;
	      sum20 = n0 * gde101 + n1 * gde111 + n2 * gde121;
	      sum3  = n0 * gde002 + n1 * gde012; // + n2 * de[2][2];
	      sum30 = n0 * gde102 + n1 * gde112; // + n2 * de0[2][2];

	      tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
	      tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
	      tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

	      tmp4 += ar * lmu * lmu0;
	      tmp5 += ar * lmu * lmu0 * dnom_1; //lmu0 * __drcp_rn(lmu + lmu0);
	      //}
	    }

	  //Scale = CUDA_LCC->jp_Scale[jp];
	  //Scale = scale; //__ldg(&CUDA_scale[bid][jp]); 
	  i = jp + (CUDA_ncoef0 - 3 + 1) * Lpoints1;

	  double * __restrict__ dytempp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;

	  /* Ders. of brightness w.r.t. rotation parameters */
	  dytempp[i] = scale * tmp1;
	  i += Lpoints1;
	  dytempp[i] = scale * tmp2;
	  i += Lpoints1;
	  dytempp[i] = scale * tmp3;
	  i += Lpoints1;

	  /* Ders. of br. w.r.t. phase function params. */
	  dytempp[i] = br * ff; //jp_dphp0; //__ldg(&jp_dphp[0][bid][jp]); 
	  i += Lpoints1;
	  dytempp[i] = br * d2; //jp_dphp1; //__ldg(&jp_dphp[1][bid][jp]); 
	  i += Lpoints1;
	  dytempp[i] = br * alpha; //jp_dphp2; //__ldg(&jp_dphp[2][bid][jp]); 

	  /* Ders. of br. w.r.t. cl, cls */
	  dytempp[jp + (CUDA_ma) * (Lpoints1) - Lpoints1] = scale * tmp4 * cl;
	  dytempp[jp + (CUDA_ma) * (Lpoints1)] = scale * tmp5;

	  /* Scaled brightness */
	  ytemp[jp] = br * scale;

	  int m, m1, iStart;
	  int d, d1, dr;

	  //if(Inrel)
	  //  {
	  iStart = 2;
	  m = bid * CUDA_Dg_block + 2 * nf1;
	  d = jp + 2 * (Lpoints1);
	  //  }
	  //else
	  //{
	  //iStart = 1;
	  //m = bid * CUDA_Dg_block + nf1;
	  //d = jp + (Lpoints1);
	  //}

	  m1 = m + nf1;

	  d1 = d + Lpoints1;
	  dr = 4 * Lpoints1;

	  /* Derivatives of brightness w.r.t. g-coeffs */
	  if(incl_count)
	    {
	      double const *__restrict__ pCUDA_Dg  = CUDA_Dg + m;
	      double const *__restrict__ pCUDA_Dg1 = CUDA_Dg + m1;
	      double const *__restrict__ pCUDA_Dg2 = CUDA_Dg + m1 + nf1;
	      double const *__restrict__ pCUDA_Dg3 = CUDA_Dg + m1 + 2 * nf1;
	      int ncoef0 = CUDA_ncoef0 - 3;

#pragma unroll 1
	      for(i = iStart; i <= ncoef0;)// i += 4, /*m += mr, m1 += mr,*/ d += dr, d1 += dr)
		{
		  double tmp = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;

		  if((i + 3) <= ncoef0)
		    {
		      j = 0;

#define UNRL16 16
#pragma unroll 2
		      for( ; j < incl_count - (UNRL16 - 1); j += UNRL16)
			{
			  double l_tmp[UNRL16], l_tmp1[UNRL16], l_tmp2[UNRL16], l_tmp3[UNRL16];
			  int l_incl[UNRL16], ii;

			  for(ii = 0; ii < UNRL16; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL16; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			      l_tmp2[ii] = (pCUDA_Dg2[l_incl[ii]]);
			      l_tmp3[ii] = (pCUDA_Dg3[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL16; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      tmp2 += qq * l_tmp2[ii];
			      tmp3 += qq * l_tmp3[ii];
			      qq = qq2;
			    }
			}

#pragma unroll 2
		      for( ; j < incl_count - (UNRL - 1); j += UNRL)
			{
			  double l_tmp[UNRL], l_tmp1[UNRL], l_tmp2[UNRL], l_tmp3[UNRL];
			  int l_incl[UNRL], ii;

			  for(ii = 0; ii < UNRL; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			      l_tmp2[ii] = (pCUDA_Dg2[l_incl[ii]]);
			      l_tmp3[ii] = (pCUDA_Dg3[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      tmp2 += qq * l_tmp2[ii];
			      tmp3 += qq * l_tmp3[ii];
			      qq = qq2;
			    }
			}
#pragma unroll 3
		      for( ; j < incl_count; j++)
			{
			  int l_incl = incl[j];
			  double l_dbr = dbr[j];
			  double v1 = (pCUDA_Dg[l_incl]);
			  double v2 = (pCUDA_Dg1[l_incl]);
			  double v3 = (pCUDA_Dg2[l_incl]);
			  double v4 = (pCUDA_Dg3[l_incl]);

			  tmp  += l_dbr * v1;
			  tmp1 += l_dbr * v2;
			  tmp2 += l_dbr * v3;
			  tmp3 += l_dbr * v4;
			}
		      __stwb(&dytempp[d], scale * tmp);
		      __stwb(&dytempp[d1], scale * tmp1);
		      __stwb(&dytempp[d1 + Lpoints1], scale * tmp2);
		      __stwb(&dytempp[d1 + 2 * Lpoints1], scale * tmp3);
		      i += 4;
		      d += dr;
		      d1 += dr;
		      pCUDA_Dg  += 4 * nf1;
		      pCUDA_Dg1 += 4 * nf1;
		      pCUDA_Dg2 += 4 * nf1;
		      pCUDA_Dg3 += 4 * nf1;
		    }
		  else if((i + 2) <= ncoef0)
		    {
#define UNRL8 8
#pragma unroll 2
		      for(j = 0 ; j < incl_count - (UNRL8 - 1); j += UNRL8)
			{
			  double l_tmp[UNRL8], l_tmp1[UNRL8], l_tmp2[UNRL8];
			  int l_incl[UNRL8], ii;

			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL8; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			      l_tmp2[ii] = (pCUDA_Dg2[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      tmp2 += qq * l_tmp2[ii];
			      qq = qq2;
			    }
			}
#pragma unroll 1
		      for( ; j < incl_count - (UNRL - 1); j += UNRL)
			{
			  double l_tmp[UNRL], l_tmp1[UNRL], l_tmp2[UNRL];
			  int l_incl[UNRL], ii;

			  for(ii = 0; ii < UNRL; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			      l_tmp2[ii] = (pCUDA_Dg2[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      tmp2 += qq * l_tmp2[ii];
			      qq = qq2;
			    }
			}
#pragma unroll 3
		      for( ; j < incl_count; j++)
			{
			  int l_incl = incl[j];
			  double l_dbr = dbr[j];
			  double v1 = (pCUDA_Dg[l_incl]);
			  double v2 = (pCUDA_Dg1[l_incl]);
			  double v3 = (pCUDA_Dg2[l_incl]);

			  tmp  += l_dbr * v1;
			  tmp1 += l_dbr * v2;
			  tmp2 += l_dbr * v3;
			}
		      __stwb(&dytempp[d], scale * tmp);
		      __stwb(&dytempp[d1], scale * tmp1);
		      __stwb(&dytempp[d1 + Lpoints1], scale * tmp2);
		      i += 3;
		      d += 3 * Lpoints1;
		      d1 += 3 * Lpoints1;
		      pCUDA_Dg  += 3 * nf1;
		      pCUDA_Dg1 += 3 * nf1;
		      pCUDA_Dg2 += 3 * nf1;
		    }
		  else if((i + 1) <= ncoef0)
		    {
#define UNRL8 8
#pragma unroll 2
		      for(j = 0 ; j < incl_count - (UNRL8 - 1); j += UNRL8)
			{
			  double l_tmp[UNRL8], l_tmp1[UNRL8];
			  int l_incl[UNRL8], ii;

			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL8; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      qq = qq2;
			    }
			}
#pragma unroll 1
		      for( ; j < incl_count - (UNRL - 1); j += UNRL)
			{
			  double l_tmp[UNRL], l_tmp1[UNRL];
			  int l_incl[UNRL], ii;

			  for(ii = 0; ii < UNRL; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }
			  double qq = dbr[j];
			  for(ii = 0; ii < UNRL; ii++)
			    { 
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]); 
			      l_tmp1[ii] = (pCUDA_Dg1[l_incl[ii]]);
			    }
			  for(ii = 0; ii < UNRL; ii++)
			    {
			      double qq2 = dbr[j + ii + 1];
			      tmp  += qq * l_tmp[ii];
			      tmp1 += qq * l_tmp1[ii];
			      qq = qq2;
			    }
			}
#pragma unroll 3
		      for( ; j < incl_count; j++)
			{
			  int l_incl = incl[j];
			  double l_dbr = dbr[j];
			  double v1 = (pCUDA_Dg[l_incl]);
			  double v2 = (pCUDA_Dg1[l_incl]);

			  tmp  += l_dbr * v1;
			  tmp1 += l_dbr * v2;
			}
		      __stwb(&dytempp[d], scale * tmp);
		      __stwb(&dytempp[d1], scale * tmp1);
		      i += 2;
		      d += 2 * Lpoints1;
		      d1 += 2 * Lpoints1;
		      pCUDA_Dg  += 2 * nf1;
		      pCUDA_Dg1 += 2 * nf1;
		    }
		  else
		    {
#define UNRL8 8
#pragma unroll 1
		      for(j = 0; j < incl_count - (UNRL8 - 1); j += UNRL8)
			{
			  double l_dbr[UNRL8], l_tmp[UNRL8];
			  int l_incl[UNRL8], ii;

			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }

			  for(ii = 0; ii < UNRL8; ii++)
			    {
			      l_dbr[ii]  = dbr[j + ii];
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]);
			    }

			  //for(ii = 0; ii < UNRL8; ii++)
			  tmp  += l_dbr[0] * l_tmp[0];
			  tmp1 += l_dbr[1] * l_tmp[1];
			  tmp2 += l_dbr[2] * l_tmp[2];
			  tmp3 += l_dbr[3] * l_tmp[3];
			  tmp  += l_dbr[4] * l_tmp[4];
			  tmp1 += l_dbr[5] * l_tmp[5];
			  tmp2 += l_dbr[6] * l_tmp[6];
			  tmp3 += l_dbr[7] * l_tmp[7];
			}
#pragma unroll 1
		      for( ; j < incl_count - (UNRL - 1); j += UNRL)
			{
			  double l_dbr[UNRL], l_tmp[UNRL];
			  int l_incl[UNRL], ii;

			  for(ii = 0; ii < UNRL; ii++)
			    {
			      l_incl[ii] = incl[j + ii];
			    }

			  for(ii = 0; ii < UNRL; ii++)
			    {
			      l_dbr[ii]  = dbr[j + ii];
			      l_tmp[ii]  = (pCUDA_Dg[l_incl[ii]]);
			    }

			  //for(ii = 0; ii < UNRL; ii++)
			  //  tmp += l_dbr[ii] * l_tmp[ii];
			  tmp  += l_dbr[0] * l_tmp[0];
			  tmp1 += l_dbr[1] * l_tmp[1];
			  tmp2 += l_dbr[2] * l_tmp[2];
			  tmp3 += l_dbr[3] * l_tmp[3];
			}
		      tmp  += tmp1;
		      tmp2 += tmp3;
#pragma unroll 3
		      for( ; j < incl_count; j++)
			{
			  int l_incl = incl[j];
			  double l_dbr = dbr[j];

			  tmp += l_dbr * (pCUDA_Dg[l_incl]);
			}
		      tmp += tmp2;
		      __stwb(&dytempp[d], scale * tmp);
		      i += 1;
		      d += 1 * Lpoints1;
		      //d1 += 1 * Lpoints1;
		      pCUDA_Dg  += nf1;
		      //pCUDA_Dg1 += nf1;
		    }
		}
	    }
	  else
	    {
	      int ncoef0 = CUDA_ncoef0 - 3;
	      double * __restrict__ p = dytempp + d;
#pragma unroll 
	      for(i = 1; i <= ncoef0 - (UNRL - 1); i += UNRL)
		for(int t = 0; t < UNRL; t++, p += Lpoints1)
		  __stwb(p, 0.0);
#pragma unroll       
	      for(; i <= ncoef0; i++, p += Lpoints1)
		__stwb(p, 0.0);
	    }



	  n += CUDA_BLOCK_DIM;
	}

    }
  //__syncwarp();

  if(Inrel == 1)
    {
      int ma = CUDA_ma;
      double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
      double const * __restrict__ pp = &(dytemp[2 * Lpoints1 + threadIdx.x]); // good, consecutive
      int bid = blockIdx();
#pragma unroll 1
      for(int i = 2; i <= ma; i++) 
        {
	  double dl = 0, dl2 = 0;
	  int nn = threadIdx.x;
	  double const *  __restrict__ p = pp;
	  
	  while(nn < Lpoints - 3*CUDA_BLOCK_DIM)
	    {
	      dl  += p[0] + p[2*CUDA_BLOCK_DIM];
	      dl2 += p[CUDA_BLOCK_DIM] + p[3*CUDA_BLOCK_DIM];
	      p   += 4 * CUDA_BLOCK_DIM;
	      nn  += 4 * CUDA_BLOCK_DIM;
	    }
#pragma unroll 2
	  while(nn < Lpoints - CUDA_BLOCK_DIM)
	    {
	      dl  += p[0];
	      dl2 += p[CUDA_BLOCK_DIM];
	      p   += 2 * CUDA_BLOCK_DIM;
	      nn  += 2 * CUDA_BLOCK_DIM;
	    }
	  //#pragma unroll 1
	  if(nn < Lpoints)
	    {
	      dl += p[0];
	      //p  += CUDA_BLOCK_DIM;
	      //nn += CUDA_BLOCK_DIM;
	    }
	  
	  dl += dl2;
	  
	  dl += __shfl_down_sync(0xffffffff, dl, 16);
	  dl += __shfl_down_sync(0xffffffff, dl, 8);
	  dl += __shfl_down_sync(0xffffffff, dl, 4);
	  dl += __shfl_down_sync(0xffffffff, dl, 2);
	  dl += __shfl_down_sync(0xffffffff, dl, 1);
	  
	  pp += Lpoints1;
	  
	  if(threadIdx.x == 0)
	    dave[bid][i - 1] = dl;
	}
      
      double d = 0, d2 = 0;
      int n = threadIdx.x;
      double const * __restrict__ p2 = &(ytemp[n]);

      while(n < Lpoints - 3*CUDA_BLOCK_DIM)
	{
	  d  += p2[0] + p2[2*CUDA_BLOCK_DIM];
	  d2 += p2[CUDA_BLOCK_DIM] + p2[3*CUDA_BLOCK_DIM];
	  p2 += 4 * CUDA_BLOCK_DIM;
	  n  += 4 * CUDA_BLOCK_DIM;
	}
#pragma unroll 2
      while(n < Lpoints - CUDA_BLOCK_DIM)
	{
	  d  += p2[0];
	  d2 += p2[CUDA_BLOCK_DIM];
	  p2 += 2 * CUDA_BLOCK_DIM;
	  n  += 2 * CUDA_BLOCK_DIM;
	}

      if(n < Lpoints)
	{
	  d += p2[0];
	}
      d += d2;
      
      d += __shfl_down_sync(0xffffffff, d, 16);
      d += __shfl_down_sync(0xffffffff, d, 8);
      d += __shfl_down_sync(0xffffffff, d, 4);
      d += __shfl_down_sync(0xffffffff, d, 2);
      d += __shfl_down_sync(0xffffffff, d, 1);

      lave = d;
    }
  
  if(threadIdx.x == 0)
    {
      int lnp = npg[0][bid];
      //aveg[bid] = lave;
      raveg[bid] = __drcp_rn(lave);
      npg[0][bid]  = lnp + Lpoints;
    }
  __syncwarp();
}



__device__ void __forceinline__  mrqcof_curve1_lastI1(freq_context * __restrict__ CUDA_LCC, int bid)
{
  int Lpoints = 3;
  int Lpoints1 = Lpoints + 1;
  int jp, lnp;
  double ymod, lave;
  __shared__ double dyda[BLOCKX4][N80];
  double * __restrict__ dydap = dyda[threadIdx.y];
  //int bid = blockIdx();
  
  lnp = npg[0][bid];

  int n = threadIdx.x, ma = CUDA_ma;
  double * __restrict__ p = &(dave[bid][n]);
#pragma unroll 2
  while(n < ma)
    {
      *p = 0;
      p += CUDA_BLOCK_DIM;
      n += CUDA_BLOCK_DIM;
    }
  lave = 0;

  //__syncthreads();

  double * __restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;
  long int lpadd = sizeof(double) * Lpoints1;

#pragma unroll 1
  for(jp = 0; jp < Lpoints; jp++)
    {
      ymod = conv(CUDA_LCC, jp, dydap, bid); 

      lnp++;
      
      if(threadIdx.x == 0)
	{
	  ytemp[jp] = ymod;
	  lave = lave + ymod;
	}
      
      int n = threadIdx.x;
      double const * __restrict__ a;
      double * __restrict__ b, * __restrict__ c;

      a = &(dydap[n]);
      b = &(dave[bid][n]);
      c = &(dytemp[jp + Lpoints1 * (n + 1)]); //ZZZ bad store order, strided

      //unrl2
#pragma unroll 2
      while(n < ma - CUDA_BLOCK_DIM)
	{ /////////////  ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZz
	  double d = a[0], bb = b[0];
	  double d2 = __ldca(&a[CUDA_BLOCK_DIM]), bb2 = __ldca(&b[CUDA_BLOCK_DIM]);

	  c[0] = d;
	  c = (double *)(((char *)c) + lpadd);
	  b[0] = bb + d;

	  c[0] = d2;
	  c = (double *)(((char *)c) + lpadd);
	  b[CUDA_BLOCK_DIM] = bb2 + d2;

	  n += 2 * CUDA_BLOCK_DIM;
	  a += 2 * CUDA_BLOCK_DIM;
	  b += 2 * CUDA_BLOCK_DIM;
	}
      //#pragma unroll 1
      if(n < ma)
	{
	  double d = a[0], bb = b[0];
	  c[0] = d;
	  b[0] = bb + d;
	}
    } /* jp, lpoints */
  
  if(threadIdx.x == 0)
    {
      npg[0][bid]  = lnp;
      raveg[bid] = __drcp_rn(lave);
    }
  
  /* save lightcurves */
  //__syncwarp();
}


__device__ void __forceinline__ mrqcof_curve1_lastI0(freq_context * __restrict__ CUDA_LCC, int bid)
{
  int Lpoints = 3;
  int Lpoints1 = Lpoints + 1;
  int jp, lnp;
  double ymod;
  __shared__ double dyda[BLOCKX4][N80];

  double * __restrict__ dydap = dyda[threadIdx.y];
  
  lnp = npg[0][bid];

  int ma = CUDA_ma;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 1
  for(jp = 0; jp < Lpoints; jp++)
    {
      lnp++;
      
      ymod = conv(CUDA_LCC, jp, dydap, bid); 
      
      if(threadIdx.x == 0)
	ytemp[jp] = ymod;
      
      int n = threadIdx.x;
      double * __restrict__ p = &dytemp[jp + Lpoints1 * (n + 1)]; // ZZZ bad store order, strided
#pragma unroll 2
      while(n < ma - CUDA_BLOCK_DIM)
	{
	  double d  = dydap[n];
	  double d2 = dydap[n + CUDA_BLOCK_DIM];
	  *p = d; //  YYYY
	  p += Lpoints1 * CUDA_BLOCK_DIM;
	  *p = d2;

	  p += Lpoints1 * CUDA_BLOCK_DIM;
	  n += 2 * CUDA_BLOCK_DIM;
	}

      if(n < ma)
	{
	  double d = dydap[n];
	  *p = d;
	}
    } /* jp, lpoints */
  
  if(threadIdx.x == 0)
    {
      npg[0][bid]  = Lpoints; //lnp;
    }
  
  /* save lightcurves */
  //__syncthreads();
}


// COF end




// conv
__device__ double __forceinline__ conv(freq_context * __restrict__ CUDA_LCC, int nc, double * __restrict__ dyda, int bid)
{
  int i, j;
  //__shared__ double res[CUDA_BLOCK_DIM];
  double tmp, tmp2; //, dtmp, dtmp2;
  int nf = CUDA_Numfac;
  int nf1 = nf + 1, nco = CUDA_Ncoef;

  j = bid * nf1 + threadIdx.x + 1;
  int xx = threadIdx.x;
  tmp = 0, tmp2 = 0;
  //double * __restrict__ areap = CUDA_Area + j;
  double const * __restrict__ areap = &(Areag[bid][threadIdx.x]);
  double * __restrict__ norp  = &(CUDA_Nor[nc][xx]);  
#pragma unroll 4
  while(xx < nf - CUDA_BLOCK_DIM)
    { 
      double a0, a1, n0, n1;
      a0 = __ldca(&areap[0]);
      n0 = __ldca(&norp[0]);
      a1 = __ldca(&areap[CUDA_BLOCK_DIM]);
      n1 = __ldca(&norp[CUDA_BLOCK_DIM]);
      areap += 2 * CUDA_BLOCK_DIM;
      norp  += 2 * CUDA_BLOCK_DIM;
      xx += 2 * CUDA_BLOCK_DIM;
      tmp += a0 * n0; //areap[0] * norp[0];
      tmp2 += a1 * n1; //areap[CUDA_BLOCK_DIM] * norp[CUDA_BLOCK_DIM];
    }
  //#pragma unroll 1
  if(xx < nf)
    {
      tmp += __ldca(&areap[0]) * __ldca(&norp[0]); 
    }

  tmp += tmp2;

  tmp += __shfl_down_sync(0xffffffff, tmp, 16);
  tmp += __shfl_down_sync(0xffffffff, tmp, 8);
  tmp += __shfl_down_sync(0xffffffff, tmp, 4);
  tmp += __shfl_down_sync(0xffffffff, tmp, 2);
  tmp += __shfl_down_sync(0xffffffff, tmp, 1);

  int ma = CUDA_ma, dg_block = CUDA_Dg_block;
  double * __restrict__ dg = CUDA_Dg, * __restrict__ darea = CUDA_Darea, * __restrict__ nor = CUDA_Nor[nc];
#pragma unroll 1
  for(j = 1; j <= ma; j++)
    {
      int m = blockIdx() * dg_block + j * nf1;
      double dtmp = 0, dtmp2 = 0; 
      if(j <= nco)
	{
	  int mm = m + threadIdx.x + 1;

	  i = threadIdx.x + 1;
	  double * __restrict__ dgp = dg + mm;
	  double * __restrict__ dareap = darea + i - 1;
	  double * __restrict__ norp = nor + i - 1;
	    
#pragma unroll 4
	  while(i <= nf - CUDA_BLOCK_DIM)
	    {
	      double g0, g1, a0, a1, n0, n1;
	      g0 = __ldca(&dgp[0]);
	      a0 = __ldca(&dareap[0]);
	      g1 = __ldca(&dgp[CUDA_BLOCK_DIM]);
	      a1 = __ldca(&dareap[CUDA_BLOCK_DIM]);
	      dgp += 2 * CUDA_BLOCK_DIM;
	      dareap += 2 * CUDA_BLOCK_DIM;
	      n0 = __ldca(&norp[0]);
	      n1 = __ldca(&norp[CUDA_BLOCK_DIM]);
	      i += 2 * CUDA_BLOCK_DIM;
	      norp += 2 * CUDA_BLOCK_DIM;
	      dtmp  += (g0 * a0) * n0;
	      dtmp2 += (g1 * a1) * n1;
	    }
	  //#pragma unroll 1
	  if(i <= nf) //; i += CUDA_BLOCK_DIM, mm += CUDA_BLOCK_DIM)
	    {
	      dtmp  += __ldca(&dgp[0]) * __ldca(&dareap[0]) * __ldca(&norp[0]); //CUDA_Dg[mm] * CUDA_Darea[i] * CUDA_Nor[nc][i];
	    }

	  dtmp += dtmp2;
	  
	  dtmp += __shfl_down_sync(0xffffffff, dtmp, 16);
	  dtmp += __shfl_down_sync(0xffffffff, dtmp, 8);
	  dtmp += __shfl_down_sync(0xffffffff, dtmp, 4);
	  dtmp += __shfl_down_sync(0xffffffff, dtmp, 2);
	  dtmp += __shfl_down_sync(0xffffffff, dtmp, 1);
	}

      if(threadIdx.x == 0)
	dyda[j-1] = dtmp;
    }

  return (tmp);
}

// conv end
/*
#define SWAP(a,b) {temp=__ldca(&(a));(a)=__ldca(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++) x[t1]=__ldca(&((a)[t1]));for(int r1=0;r1<4;r1++) y[r1]=__ldca(&((b)[r1]));for(int t2=0;t2<4;t2++)(b)[t2]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP8(a,b) {double x[8];for(int t1=0;t1<8;t1++) x[t1]=__ldca(&((a)[t1]));for(int t2=0;t2<8;t2++)(a)[t2]=__ldca(&((b)[t2]));for(int t3=0;t3<8;t3++)(b)[t3]=x[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=__ldca(&((a)[t1*n]));for(int r1=0;r1<4;r1++)y[r1]=__ldca(&((b)[r1*n]));for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#define SWAP8n(a,b,n) {double x[8];for(int t1=0;t1<8;t1++)x[t1]=__ldca(&((a)[t1*n]));for(int t2=0;t2<8;t2++)(a)[t2*n]=__ldca(&((b)[t2*n]));for(int t3=0;t3<8;t3++)(b)[t3*n]=x[t3];}

#define SWAP(a,b) {temp=__ldg(&(a));(a)=__ldca(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++) x[t1]=__ldg(&((a)[t1]));for(int r1=0;r1<4;r1++) y[r1]=__ldca(&((b)[r1]));for(int t2=0;t2<4;t2++)(b)[t2]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP8(a,b) {double x[8];for(int t1=0;t1<8;t1++) x[t1]=__ldg(&((a)[t1]));for(int t2=0;t2<8;t2++)(a)[t2]=__ldca(&((b)[t2]));for(int t3=0;t3<8;t3++)(b)[t3]=x[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=__ldg(&((a)[t1*n]));for(int r1=0;r1<4;r1++)y[r1]=__ldg(&((b)[r1*n]));for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#define SWAP8n(a,b,n) {double x[8];for(int t1=0;t1<8;t1++)x[t1]=__ldg(&((a)[t1*n]));for(int t2=0;t2<8;t2++)(a)[t2*n]=__ldg(&((b)[t2*n]));for(int t3=0;t3<8;t3++)(b)[t3*n]=x[t3];}
*/
// GAUSS
//#define SWAP_NORMAL
//#define SWAP_NORMAL_V2
//#define SWAP_NORMAL_V3
#define SWAP_NORMAL_V4
//#define SWAP_LDCA
//#define SWAP_LDCA_V2
//#define SWAP_LDG
//#define SWAP_LDG_V2

#ifdef SWAP_NORMAL
#define SWAP(a,b) {double temp=(a);(a)=(b);(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=(a)[t1];for(int r1=0;r1<4;r1++)y[r1]=(b)[r1];for(int t2=0;t2<4;t2++)(b)[t2]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=(a)[t1*n];for(int r1=0;r1<4;r1++)y[r1]=(b)[r1*n];for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#endif

#ifdef SWAP_NORMAL_V2
#define SWAP(a,b) {double temp=(a);(a)=(b);(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=(a)[t1];y[t1]=(b)[t1];}for(int t2=0;t2<4;t2++){(b)[t2]=x[t2];(a)[t2]=y[t2];}}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=(a)[t1*n];y[t1]=(b)[t1*n];}for(int t2=0;t2<4;t2++){(b)[t2*n]=x[t2];(a)[t2*n]=y[t2];}}
#endif

#ifdef SWAP_NORMAL_V3
#define SWAP(a,b) {double temp=*(const double * __restrict__)&(a);(a)=*(const double *__restrict__)&(b);(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=*(const double * __restrict__)&((a)[t1]);y[t1]=*(const double * __restrict__)&((b)[t1]);}for(int t2=0;t2<4;t2++){(b)[t2]=x[t2];(a)[t2]=y[t2];}}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=*(const double * __restrict__)&((a)[t1*n]);y[t1]=*(const double * __restrict__)&((b)[t1*n]);}for(int t2=0;t2<4;t2++){(b)[t2*n]=x[t2];(a)[t2*n]=y[t2];}}
#endif

#ifdef SWAP_NORMAL_V4
#define SWAP(a,b) {double const * __restrict__ aa=&(a); double const * __restrict__ bb = &(b); double temp=*aa;(a)=*bb;(b)=temp;}
#define SWAP4(a,b) {double const * __restrict__ aa = (a);double const * __restrict__ bb = (b);double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=(aa[t1]);y[t1]=__ldca(&((bb)[t1]));}for(int t2=0;t2<4;t2++){(b)[t2]=x[t2];(a)[t2]=y[t2];}}
#define SWAP4n(a,b,n) {double const * __restrict__ aa = (a); double const * __restrict__ bb = (b);double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=(aa[t1*n]);y[t1]=__ldca(&bb[t1*n]);}for(int t2=0;t2<4;t2++){(b)[t2*n]=x[t2];(a)[t2*n]=y[t2];}}
#endif

#ifdef SWAP_LDCA
#define SWAP(a,b) {double temp=__ldca(&(a));(a)=__ldca(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++) x[t1]=__ldca(&((a)[t1]));for(int r1=0;r1<4;r1++) y[r1]=__ldca(&((b)[r1]));for(int t2=0;t2<4;t2++)(b)[t2]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=__ldca(&((a)[t1*n]));for(int r1=0;r1<4;r1++)y[r1]=__ldca(&((b)[r1*n]));for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#endif

#ifdef SWAP_LDCA_V2
#define SWAP(a,b) {double temp=__ldca(&(a));(a)=__ldca(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=__ldca(&((a)[t1]));y[t1]=__ldca(&((b)[t1]));}for(int t2=0;t2<4;t2++){(b)[t2]=x[t2];(a)[t2]=y[t2];}}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=__ldca(&((a)[t1*n]));y[t1]=__ldca(&((b)[t1*n]));}for(int t2=0;t2<4;t2++){(b)[t2*n]=x[t2];(a)[t2*n]=y[t2];}}
#endif

#ifdef SWAP_LDG
#define SWAP(a,b) {double temp=__ldg(&(a));(a)=__ldg(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=__ldg(&((a)[t1]));for(int r1=0;r1<4;r1++)y[r1]=__ldg(&((b)[r1]));for(int t2=0;t2<4;t2++)(b)[t2]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=__ldg(&((a)[t1*n]));for(int r1=0;r1<4;r1++)y[r1]=__ldg(&((b)[r1*n]));for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#endif

#ifdef SWAP_LDG_V2
#define SWAP(a,b) {double temp=__ldg(&(a));(a)=__ldg(&(b));(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=__ldg(&((a)[t1]));y[t1]=__ldg(&((b)[t1]));}for(int t2=0;t2<4;t2++){(b)[t2]=x[t2];(a)[t2]=y[t2];}}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++){x[t1]=__ldg(&((a)[t1*n]));y[t1]=__ldg(&((b)[t1*n]));}for(int t2=0;t2<4;t2++){(b)[t2*n]=x[t2];(a)[t2*n]=y[t2];}}
#endif

//#define SWAP8(a,b) {double x[8];for(int t1=0;t1<8;t1++) x[t1]=(a)[t1];for(int t2=0;t2<8;t2++)(a)[t2]=(b)[t2];for(int t3=0;t3<8;t3++)(b)[t3]=x[t3];}
//#define SWAP8n(a,b,n) {double x[8];for(int t1=0;t1<8;t1++)x[t1]=(a)[t1*n];for(int t2=0;t2<8;t2++)(a)[t2*n]=(b)[t2*n];for(int t3=0;t3<8;t3++)(b)[t3*n]=x[t3];}

__device__ int __forceinline__ gauss_errc(freq_context * __restrict__ CUDA_LCC, int ma)
{
  __shared__ int16_t sh_icol[N80]; //[CUDA_BLOCK_DIM];
  __shared__ int16_t sh_irow[N80]; //[CUDA_BLOCK_DIM];
  __shared__ int16_t indxr[N80]; //[MAX_N_PAR + 1];
  __shared__ int16_t indxc[N80]; //[MAX_N_PAR + 1];
  __shared__ int16_t ipiv[N80];  //[MAX_N_PAR + 1];
  __shared__ double pivinv;
  __shared__ int icol;
  __shared__ double sh_big[N80]; //[CUDA_BLOCK_DIM];

  int mf  = CUDA_mfit;
  int mf1 = mf + 1;
  
  int j = threadIdx.x + 1;

#pragma unroll 4
  while(j <= mf - CUDA_BLOCK_DIM)
    {
      ipiv[j] = 0;
      ipiv[j + CUDA_BLOCK_DIM] = 0;
      j += 2 * CUDA_BLOCK_DIM;
    }
  if(j <= mf)
    {
      ipiv[j] = 0;
      //j += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  double const * __restrict__ covarp = CUDA_LCC->covar;

#pragma unroll 1
  for(int i = 1; i <= mf; i++)
    {
      double big = 0.0;
      int irow = 0;
      int licol = 0;
      int j = threadIdx.x + 1;

#pragma unroll 1
      while(j <= mf)
	{
	  if(ipiv[j] != 1)
	    {
	      int ixx = j * mf1 + 1;
#pragma unroll 4
	      for(int k = 1; k <= mf; k++, ixx++)
		{
		  int ii = ipiv[k];
		  if(ii == 0)
		    {
		      double tmpcov = fabs(__ldca(&covarp[ixx]));
		      if(tmpcov >= big)
			{
			  irow = j;
			  licol = k;
			  big = tmpcov;
			}
		    }
		  else if(ii > 1)
		    {
		      return(1);
		    }
		}
	    }
	  j += CUDA_BLOCK_DIM;
	}

      int jj = threadIdx.x;
#pragma unroll 4
      while(jj <= mf - CUDA_BLOCK_DIM)
	{      
	  sh_big[jj] = big;
	  sh_irow[jj] = irow;
	  sh_icol[jj] = licol;
	  sh_big[jj + CUDA_BLOCK_DIM] = big;
	  sh_irow[jj + CUDA_BLOCK_DIM] = irow;
	  sh_icol[jj + CUDA_BLOCK_DIM] = licol;
	  jj += 2 * CUDA_BLOCK_DIM;
	}
      if(jj <= mf)
	{      
	  sh_big[jj] = big;
	  sh_irow[jj] = irow;
	  sh_icol[jj] = licol;
	  jj += CUDA_BLOCK_DIM;
	}
      
      __syncwarp();
      
      if(threadIdx.x == 0)
	{
	  big = sh_big[0];
	  icol = sh_icol[0];
	  irow = sh_irow[0];
#pragma unroll 2
	  for(int j = 1; j <= mf; j++)
	    {
	      if(sh_big[j] >= big)
		{
		  big = sh_big[j];
		  irow = sh_irow[j];
		  icol = sh_icol[j];
		}
	    }

	  ipiv[icol]++;

	  double * __restrict__ dapp = CUDA_LCC->da;

	  if(irow != icol)
	    {
	      double * __restrict__ cvrp = (double *)covarp + irow * mf1; 
	      double * __restrict__ cvcp = (double *)covarp + icol * mf1;
	      int l;

#pragma unroll 6
	      for(l = 1; l <= mf - 3; l += 4)
		{
		  SWAP4(cvrp, cvcp);
		  cvrp += 4;
		  cvcp += 4;
		}
	      
#pragma unroll 3
	      for(; l <= mf; l++)
		{
		  SWAP(cvrp[0], cvcp[0]);
		  cvrp++;
		  cvcp++;
		}
	      
	      SWAP(dapp[irow], dapp[icol]);
	    }

	  indxr[i] = irow;
	  indxc[i] = icol;
	  double cov = __ldca(&covarp[icol * mf1 + icol]);

	  if(cov == 0.0) 
	    {
	      int bid = blockIdx();
	      
	      int    const * __restrict__ iap = CUDA_ia + 1;
	      double * __restrict__ atp = atry[bid] + 1; 
	      double * __restrict__ cgp = cgg[bid] + 1; 
	      double * __restrict__ dap = dapp;

#pragma unroll 8
	      for(int l = 0; l < ma; l++)
		{
		  if(*iap)
		    {
		      dap++;
		      __stwb(atp,  __ldca(cgp) + __ldca(dap));
		    }
		  iap++;
		  atp++;
		  cgp++;
		}
	      
	      return(2);
	    }

	  pivinv = __drcp_rn(cov);
	  double * __restrict ppp = (double *)covarp;
	  ppp[icol * mf1 + icol] = 1.0;
	  dapp[icol] *= pivinv;
	}
      
      __syncwarp();
      
      int x = threadIdx.x + 1;
      double * __restrict__ p = (double *)&covarp[icol * mf1];
#pragma unroll 1
      while(x <= mf - CUDA_BLOCK_DIM)
	{
	  __stwb(&p[x], p[x] * pivinv);
	  __stwb(&p[x + CUDA_BLOCK_DIM], p[x+CUDA_BLOCK_DIM] * pivinv);
	  x += 2*CUDA_BLOCK_DIM;
	}
      if(x <= mf)
	{
	  __stwb(&p[x], p[x] * pivinv);
	  x += CUDA_BLOCK_DIM;
	}
      
      __syncwarp();

      double *dapp = CUDA_LCC->da;
#pragma unroll 2
      for(int ll = 1; ll <= mf; ll++)
	if(ll != icol)
	  {
	    int ixx = ll * mf1, jxx = icol * mf1;
	    double dum = covarp[ixx + icol];
	    __stwb((double *)&covarp[ixx + icol], 0.0);
	    ixx++;
	    jxx++;
	    ixx += threadIdx.x;
	    jxx += threadIdx.x;
	    int l = threadIdx.x;
#pragma unroll 2
	    while(l < mf - CUDA_BLOCK_DIM)
	      {
		__stwb((double *)&covarp[ixx],  covarp[ixx] - covarp[jxx] * dum);
		__stwb((double *)&covarp[ixx + CUDA_BLOCK_DIM],  covarp[ixx+CUDA_BLOCK_DIM] - covarp[jxx+CUDA_BLOCK_DIM] * dum);
		l += 2*CUDA_BLOCK_DIM;
		ixx += 2*CUDA_BLOCK_DIM;
		jxx += 2*CUDA_BLOCK_DIM;
	      }

	    if(l < mf)
	      {
		__stwb((double *)&covarp[ixx],  covarp[ixx] - covarp[jxx] * dum);
		l += CUDA_BLOCK_DIM;
		ixx += CUDA_BLOCK_DIM;
		jxx += CUDA_BLOCK_DIM;
	      }

	    __stwb(&dapp[ll], dapp[ll] - dapp[icol] * dum);
	  }
      
      __syncwarp();
    }

  int l = mf - threadIdx.x;
#pragma unroll 2
  while(l >= 1)
    {
      int r = indxr[l];
      int c = indxc[l];
      if(r != c)
	{
	  double * __restrict__ cvp1 = (double *)&(covarp[0]), * __restrict__ cvp2;
	  cvp2 = cvp1;
	  int i1 = mf1 + r;
	  int i2 = mf1 + c;
	  cvp1 = cvp1 + i1;
	  cvp2 = cvp2 + i2;
	  int k;
	  
#pragma unroll 6
	  for(k = 1; k <= mf - 3; k += 4)
	    {
	      SWAP4n(cvp1, cvp2, mf1);
	      cvp1 += mf1 * 4;
	      cvp2 += mf1 * 4;
	    }
#pragma unroll 3
	  for(; k <= mf; k++)
	    {
	      SWAP(cvp1[0], cvp2[0]);
	      cvp1 += mf1;
	      cvp2 += mf1;
	    }
	}
      l -= CUDA_BLOCK_DIM;
    }

  __syncwarp();

  return(0);
}
#undef SWAP
/* from Numerical Recipes */

// GAUSS ends

// curve2 variants begin
// Some of them is/are NN% of total run time




__device__ void __forceinline__ MrqcofCurve2I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  //inrel = 0;
  int l, jp, j, /*k, m,*/ lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  int mf1 = CUDA_mfit + 1;
  
  __shared__ double dydat[4][N80];
  
  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 2
  for(jp = 0; jp < lpoints; jp++)
    {
      if(((jp)&3) == 0)
	{
	  int tid = threadIdx.x >> 2;
	  int u = threadIdx.x & 3;
	  int ixx = jp + (tid + 1) * Lpoints1; // ZZZ bad, strided read dytemp, BAD
	  double * __restrict__ c = &(dytemp[ixx]);//, *dddc = ddd + ixx;
	  c += u;
	  l = tid;
#pragma unroll 4
	  while(l < ma)
	    {
	      dydat[u][l] = __ldca(c); //*dddc //__ldca(c); // YYYY

	      l += CUDA_BLOCK_DIM/4;
	      c += CUDA_BLOCK_DIM/4 * Lpoints1;
	    }
	}
      __syncwarp();
      
      double * __restrict__ dyda = &(dydat[(jp) & 3][0]);	  

      lnp2++;

      ymod = __ldca(&ytemp[jp]);
      sig2i = __ldg(&CUDA_sigr2[lnp2]);
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;

      //j = 0;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alph = &alpha[mf1 + threadIdx.x + 1];
      double *alphp = alph,  *alphpp = alph;
#pragma unroll 2
      for(l = 1; l < lastone; l++)
	{
	  wt = dyda[l - 1] * sig2iwght;
	  int xx = threadIdx.x;
	  alphpp += mf1;
#pragma unroll 2
	  while(xx <= l)
	    {
	      __stwb(alph, __ldca(alph) + wt * dyda[xx]); //ldg
	      alph  += CUDA_BLOCK_DIM;
	    } /* m */

	  alph = alphpp;
	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  l++;
	} /* l */
	  
      int * __restrict__ iapp = CUDA_ia;
      alph = alphp;
#pragma unroll 1
      while(l < lma)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      wt = dyda[l - 1] * sig2iwght;
	      int xx = threadIdx.x;

#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alph, __ldca(alph) + wt * dyda[xx]); //ldg
		  alph  += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  int k = lastone;
		  int m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  double * __restrict__ alp = alphp + k; //ha + l * mf1 + k;
		  beta[l] = __ldca(&beta[l]) + dy * wt;
#pragma unroll 4
		  while(m <= l)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m]);
			  alp++;
			}
		      iap++;
		      m++;
		    } /* m */
		}
	    }
	  alphp += mf1;
	  alph = alphp;
	  l++;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}


// SLOWW
__device__ void __forceinline__ MrqcofCurve2I1IA0(freq_context *__restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  int mf1 = CUDA_mfit + 1;
  __shared__ double dydat[4][N80];
  
  lnp1 = npg[1][bid] + threadIdx.x + 1;

  int ma = CUDA_ma;
  jp = threadIdx.x;
  double rave = raveg[bid]; 
  double * __restrict__ dytempp = CUDA_LCC->dytemp, * __restrict__ ytempp = CUDA_LCC->ytemp;
  double * __restrict__ cuda_sig = CUDA_sig;
  double * __restrict__ davep = &(dave[bid][0]);
  long int lpadd = sizeof(double) * Lpoints1;
  
#pragma unroll 1
  while(jp < lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytempp[ixx] = 0; // YYY, good, consecutive

      coef = __ldca(&cuda_sig[lnp1]) * lpoints * rave; 

      double yytmp = __ldca(&ytempp[jp]);
      coef1 = yytmp * rave; 
      ytempp[jp] = coef * yytmp;

      ixx += Lpoints1;
      double const * __restrict__ dyp = &(dytempp[ixx]);
      double const * __restrict__ dypp; //, *ddyp = ddd + ixx, *ddypp; 
      double const * __restrict__ dap = &(davep[1]);

#pragma unroll 1
      for(l = 2; l <= ma - (4 - 1); l += 4, ixx += 4 * Lpoints1)
	{
	  double dd[4], dy[4];
	  int ii;
	  dypp = dyp;

#pragma unroll 4
	  for(ii = 0; ii < 4; ii++)
	    {
	      dy[ii] = *dypp; //__ldca(dypp);
	      dypp = (double *)(((char *)dypp) + lpadd);
	      dd[ii] = *dap;//__ldca(dap);
	      dap++;
	    }
#pragma unroll 4
	  for(ii = 0; ii < 4; ii++)
	    {
	      double d = coef * (dy[ii] - coef1 * dd[ii]);
	      double * __restrict__ dyppp = (double *)dyp;
	      *dyppp = d;
	      dyp = (double *)(((char *)dyp) + lpadd);
	    }
	}
#pragma unroll 3
      while(l <= ma)
	{
	  double d = coef * __ldca(&dyp[0]) - coef1 * __ldca(&dap[0]);
	  double * __restrict__ dyppp = (double *)dyp;
	  *dyppp = d;
	  l++;
	  dyp = (double *)(((char *)dyp) + lpadd);
	  dap++;
	}
      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }
  
  __syncwarp();
  
  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }
  
  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];
  
  int lastone = CUDA_lastone, lma = CUDA_lastma;
  double * __restrict__ cuda_weight = CUDA_Weight, * __restrict__ cuda_brightness = CUDA_brightness;
  
#pragma unroll 4
  for(jp = 0; jp < lpoints; jp++)
    {
      if(((jp)&3) == 0)
	{
	  int tid = threadIdx.x >> 2;
	  int u = threadIdx.x & 3;
	  int ixx = jp + (tid + 1) * Lpoints1; // ZZZ bad, strided read dytemp, BAD
	  double * __restrict__ c = &(dytempp[ixx]);//, *dddc = ddd + ixx;
	  c += u;
	  l = tid;
#pragma unroll 4
	  while(l < ma)
	    {
	      dydat[u][l] = __ldca(c);
	      l += CUDA_BLOCK_DIM/4;
	      c += CUDA_BLOCK_DIM/4 * Lpoints1;
	    }
	}
      __syncwarp();
      double * __restrict__ dyda = &(dydat[(jp) & 3][0]);	  
      lnp2++;

      ymod = ytempp[jp];
      sig2i = __ldg(&CUDA_sigr2[lnp2]); 
      wght = cuda_weight[lnp2];
      dy = cuda_brightness[lnp2] - ymod;

      //j = 0;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double *__restrict__ alph = &alpha[mf1 + threadIdx.x];
      double *alphp = alph; //, *alphpp = alph;

#pragma unroll 2
      for(l = 1; l < lastone; l++)
	{
	  wt = dyda[l] * sig2iwght;

	  int xx = threadIdx.x;
#pragma unroll 2
	  while(xx <= l)
	    {
	      __stwb(alph, __ldca(alph) +  wt * dyda[xx]);
	      alph += CUDA_BLOCK_DIM;
	      xx += CUDA_BLOCK_DIM;
	    } /* m */

	  ++betap;
	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alphp += mf1;
	  alph = alphp;
	} /* l */
	  
      int * __restrict__ iapp = CUDA_ia;
      alph = alphp;
#pragma unroll 1
      for(; l < lma; l++)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      //j++;
	      wt = dyda[l - 1] * sig2iwght;

	      int xx = threadIdx.x;

#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alph, __ldca(alph) + wt * dyda[xx]);
		  alph += CUDA_BLOCK_DIM;
		  xx += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  double * __restrict__ alp = alphp + k; //&(alpha[l * mf1 + k]);
#pragma unroll 4
		  for(; m < l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m-1]);
			  ++alp;
			}
		      iap++;
		    } /* m */

		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  alphp += mf1;
	  alph = alphp;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}




__device__ void __forceinline__ MrqcofCurve2I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  int mf1 = CUDA_mfit + 1;
  __shared__ double dyda[N80];
  
  //__syncwarp(); // remove

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 2
  for(jp = 0; jp < lpoints; jp++) // CHANGE LOOP threadIdx.x ?
    {
      lnp2++;

      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;

	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;

	  dyda[l] = a;
	  dyda[l + CUDA_BLOCK_DIM] = b;

	  l += 2*CUDA_BLOCK_DIM;
	}

      if(l < ma)
	dyda[l] = __ldca(c);

      __syncwarp();

      ymod = __ldca(&(ytemp[jp]));
      sig2i = __ldg(&CUDA_sigr2[lnp2]);  
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alp = alpha + mf1 + threadIdx.x + 1;
      double *alpp = alp;

#pragma unroll 2
      for(l = 1; l < lastone; l++)
	{
	  wt = dyda[l] * sig2iwght;
	  int xx = threadIdx.x;

#pragma unroll 2
	  while(xx < l)
	    {
	      __stwb(alp, __ldca(alp) + wt * dyda[xx]);
	      alp += CUDA_BLOCK_DIM; //mf1;
	      xx += CUDA_BLOCK_DIM;
	    } /* m */

	  ++betap;
	  alpp += mf1;
	  if(threadIdx.x == 0)
	    {
	      //double *betap = beta + l;
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alp = alpp;
	} /* l */
	  
      int * __restrict__ iapp = CUDA_ia;
      alp = alpp;
#pragma unroll 2
      for(; l < lma; l++)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      wt = dyda[l] * sig2iwght;
	      int xx = threadIdx.x;
#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
		  xx += CUDA_BLOCK_DIM;
		  alp  += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m]);
			  alp++;
			}
		      iap++;
		    } /* m */
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  alpp += mf1;
	  alp = alpp;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}




// WORKING, SLOW
  __device__ void __forceinline__ MrqcofCurve2I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  int mf1 = CUDA_mfit + 1;
  __shared__ double dyda[N80];
  
  lnp1 = npg[1][bid] + threadIdx.x + 1;
  
  int ma = CUDA_ma;
  //int bid = blockIdx();
  jp = threadIdx.x;
  double rave = raveg[bid]; 
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 1
  while(jp < lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytemp[ixx] = 0; // YYY, good, consecutive
      double yytmp = ytemp[jp];
      coef = __ldg(&CUDA_sig[lnp1]) * lpoints * rave; 

      ixx += Lpoints1;
      double const * __restrict__ dyp = &(dytemp[ixx]);
      double const * __restrict__ dap = &(dave[bid][1]);

      coef1 = yytmp * rave; 
      ytemp[jp] = coef * yytmp;

#pragma unroll 2
      for(l = 2; l <= ma - (UNRL - 1); l += UNRL, ixx += UNRL * Lpoints1)
	{
	  double dd[UNRL], dy[UNRL];
	  int ii;
	  double * __restrict__ dypp = (double *)dyp;
#pragma unroll 
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      dy[ii] = *dyp; //__ldca(dyp);
	      dyp += Lpoints1;
	      dd[ii] = *dap; //__ldg(dap);
	      dap++;
	    }
#pragma unroll
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      __stwb(dypp, coef * (dy[ii] - coef1 * dd[ii]));
	      dypp += Lpoints1;
	    }
	}

#pragma unroll 3
      for(; l <= ma; l++, dyp += Lpoints1, dap++)
	{
	  double *dypp = (double *)dyp;
	  __stwb(dypp, __ldca(dyp) * coef - coef1 * __ldg(dap));
	}
      
      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int lastone = CUDA_lastone, lma = CUDA_lastma;
  
#pragma unroll 2
  for(jp = 0; jp < lpoints; jp++) // CHANGE LOOP threadIDx.x ?
    {
      lnp2++;

      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;

	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;

	  dyda[l] = a;
	  dyda[l + CUDA_BLOCK_DIM] = b;

	  l += 2*CUDA_BLOCK_DIM;
	}

      if(l < ma)
	dyda[l] = __ldca(c);
	    
      __syncwarp();

      //j = 0;
      ymod = __ldca(&(ytemp[jp]));
      sig2i = __ldg(&CUDA_sigr2[lnp2]); //__drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alp = &alpha[mf1 + threadIdx.x + 1];
      double *alpp = alp;

#pragma unroll 2
      for(l = 1; l <= lastone; l++)
	{
	  wt = dyda[l] * sig2iwght;
	  int xx = threadIdx.x;
#pragma unroll 2
	  while(xx < l)
	    {
	      __stwb(alp, __ldca(alp) + wt * dyda[xx]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } // m 

	  ++betap;
	  alpp += mf1;
	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alp = alpp;
	} // l
	  
      int * __restrict__ iapp = CUDA_ia;
      alp = alpp;

#pragma unroll 1
      for(; l < lma; l++)
	{
	  ++betap;
	  if(iapp[l])
	    {
	      //j++;
	      wt = dyda[l - 1] * sig2iwght;
	      int xx = threadIdx.x;

#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} // m 

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k;
#pragma unroll 4
		  for(; m < l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m]);
			  alp++;
			}
		      iap++;
		    } // m 
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  alpp += mf1;
	  alp = alpp;
	} // l 
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } // jp 

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}



// SLOW (only 3 threads participate -> 1/10 perf))
  __device__ void __forceinline__ MrqcofCurve23I1IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit + 1;
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  __shared__ double dydat[3][N80];
  
  lnp1 = npg[1][bid] + threadIdx.x + 1;

  int ma = CUDA_ma;
  //int bid = blockIdx();
  jp = threadIdx.x;
  double rave = raveg[bid]; 
  double * __restrict__ dytmpp = CUDA_LCC->dytemp, * __restrict__ cuda_sig = CUDA_sig, * __restrict__ ytemp = CUDA_LCC->ytemp;
  double * __restrict__ cuda_weight = CUDA_Weight, * __restrict__ cuda_brightness = CUDA_brightness;
  double * __restrict__ davep = &(dave[bid][0]);
  long int lpadd = sizeof(double) * Lpoints1;
  
  //#pragma unroll 
  if(jp < lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytmpp[ixx] = 0; // YYY, good, consecutive
      coef = cuda_sig[lnp1] * lpoints * rave; // / CUDA_LCC->ave;
      
      double yytmp = ytemp[jp];
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytemp[jp] = coef * yytmp;
      
      ixx += Lpoints1;
      double * __restrict__ dyp = dytmpp + ixx; //&(CUDA_LCC->dytemp[ixx]);
      double * __restrict__ dap = &(davep[1]);
#pragma unroll 2
      for(l = 1; l < ma - (UNRL - 1); l += UNRL) //, ixx += UNRL * Lpoints1)
	{
	  double dd[UNRL], dy[UNRL];
	  int ii;
	  double * __restrict__ dypp = dyp;
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      dy[ii] = __ldg(dypp);
	      //dypp += Lpoints1;
	      dypp = (double *)(((char *)dypp) + lpadd);

	      dd[ii] = __ldca(dap);
	      dap++;
	    }
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      __stwb(dyp, coef * (dy[ii] - coef1 * dd[ii])); //WXX
	      dyp = (double *)(((char *)dyp) + lpadd);
	    }
	}
#pragma unroll 1
      for(; l < ma; l++, dyp += Lpoints1, dap++)
	__stwb(dyp, coef * ( __ldg(dyp) - coef1 * __ldca(dap))); //WXX

      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];
  int lastone = CUDA_lastone;

#pragma unroll 
  for(jp = 0; jp < lpoints; jp++)
    {
      if(jp == 0)
	{
	  int ixx = (threadIdx.x + 1) * Lpoints1; // RXX bad, strided read, BAD
	  double * __restrict__ c = dytmpp + ixx;  //&(CUDA_LCC->dytemp[ixx]);
	  l = threadIdx.x;
#pragma unroll 2
	  while(l < ma)
	    {
	      dydat[0][l] = c[0]; // YYYY RXX
	      dydat[1][l] = c[1]; // YYYY
	      dydat[2][l] = c[2]; // YYYY
	      l += CUDA_BLOCK_DIM;
	      c += CUDA_BLOCK_DIM * Lpoints1;
	    }
	  __syncwarp();
	}

      double * __restrict__ dyda = &dydat[jp][0];

      //j = 0;
      lnp2++;
      //double s = cuda_sig[lnp2];
      ymod = ytemp[jp];
      sig2i = __ldg(&CUDA_sigr2[lnp2]); //__drcp_rn(s * s);
      wght = cuda_weight[lnp2];
      dy = cuda_brightness[lnp2] - ymod;
      
      double sig2iwght = sig2i * wght;

      double * __restrict__ dydap = dyda + 1;
      double *betap = beta;
      double * __restrict__ alp = &(alpha[mf1 + threadIdx.x + 1]);
      double *alpp = alp;
#pragma unroll 
      for(l = 1; l <= lastone; l++)
	{
	  //j++;
	  wt = *dydap * sig2iwght;
	  dydap++;
	  int xx = threadIdx.x;
	  
#pragma unroll 2
	  while(xx < l)
	    {
	      __stwb(alp, __ldca(alp) + wt * dyda[xx]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */

	  ++betap;
	  alpp += mf1;

	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alp = alpp;
	} /* l */
      
      int * __restrict__ iapp = CUDA_ia;
      alp = alpp;

#pragma unroll 
      for(; l < CUDA_lastma; l++)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      int xx = threadIdx.x;
	      wt = *dydap * sig2iwght;
	      
#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k; 
#pragma unroll 4
		  for(; m < l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m]);
			  alp++;
			}
		      iap++;
		    } /* m */
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  alpp += mf1;
	  alp = alpp;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}


  __device__ void __forceinline__ MrqcofCurve23I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit + 1;
  //int bid = blockIdx();
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  __shared__ double dyda[N80];
  
  lnp1 = npg[1][bid] + 1;

  int ma = CUDA_ma;
  double rave = raveg[bid]; //__drcp_rn(aveg[bid]);
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 
  for(jp = 0; jp < lpoints; jp++, lnp1++)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytemp[ixx] = 0; // YYY, good?, same for all threads??
      double yytmp = ytemp[jp];
      coef = __ldg(&CUDA_sig[lnp1]) * lpoints * rave; // / CUDA_LCC->ave;

      ixx += Lpoints1;
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytemp[jp] = coef * yytmp;

      double * __restrict__ dyp = &(dytemp[ixx]);
      double * __restrict__ dap = &(dave[bid][1]);
      l = 1 + threadIdx.x;
#pragma unroll 2
      while(l < ma)
	{
	  double dy = __ldg(dyp);
	  double dd = __ldca(dap);
	  dap += CUDA_BLOCK_DIM;
	  __stwb(dyp, coef * (dy - coef1 * dd));
	  dyp += Lpoints1 * CUDA_BLOCK_DIM;
	  l += CUDA_BLOCK_DIM;
	  ixx += CUDA_BLOCK_DIM * Lpoints1;
	}
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int lastone = CUDA_lastone, lma = CUDA_lastma;
  
#pragma unroll 
  for(jp = 0; jp < lpoints; jp++) 
    {
      lnp2++;
      //double s = __ldg(&CUDA_sig[lnp2]);
      
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l] = a;
	  dyda[l + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
#pragma unroll 1
      while(l <= ma)
	{
	  dyda[l] = __ldca(c);
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      //j = 0;
      ymod = __ldca(&(ytemp[jp]));
      sig2i = __ldg(&CUDA_sigr2[lnp2]); //__drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alp = &alpha[mf1 + threadIdx.x + 1];
      double *alpp = alp;
#pragma unroll 4
      for(l = 1; l < lastone; l++)
	{
	  wt = dyda[l] * sig2iwght;
	  int xx = threadIdx.x;

#pragma unroll 2
	  while(xx <= l)
	    {
	      __stwb(alp, __ldca(alp) +  wt * dyda[xx]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */

	  ++betap;
	  alpp += mf1;

	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  
	  alp = alpp;
	} /* l */
      
      int * __restrict__ iapp = CUDA_ia;
      alp = alpp;
#pragma unroll 4
      while(l < lma)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      int xx = threadIdx.x;
	      wt = dyda[l - 1] * sig2iwght;

#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k;
#pragma unroll 4
		  while(m <= l)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[m]);
			  alp++;
			}
		      iap++;
		      m++;
		    } /* m */
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  l++;
	  alpp += mf1;
	  alp = alpp;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */
  
  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}
  
  
  
__device__ void __forceinline__ MrqcofCurve23I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit + 1;
  int l, jp, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  __shared__ double dyda[BLOCKX4][N80];
  double * __restrict__ dydap = dyda[threadIdx.y];
  //__syncthreads();

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 
  for(jp = 0; jp < lpoints; jp++)
    {
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ bad, strided read, BAD
      double * __restrict__ c = &(dytemp[ixx]);
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma)
	{
	  dydap[l] = __ldca(c); // YYYY
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      lnp2++;
      //double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&ytemp[jp]);
      sig2i = __ldg(&CUDA_sigr2[lnp2]); //__drcp_rn(s * s);
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      
      //j = 0;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alp = &alpha[mf1 + threadIdx.x];
      double *alpp = alp;

#pragma unroll 
      for(l = 1; l < lastone; l++)
	{
	  int xx = threadIdx.x;
	  wt = dydap[l] * sig2iwght;

#pragma unroll 2
	  while(xx <= l)
	    {
	      __stwb(alp, __ldca(alp) + wt * dydap[xx]);
	      xx  += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */

	  ++betap;
	  alpp += mf1;
	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alp = alpp;
	} /* l */
      
      int * __restrict__ iapp = CUDA_ia;

      alp = alpp;
#pragma unroll 
      for(; l < lma; l++)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      int xx = threadIdx.x;
	      wt = dydap[l-1] * sig2iwght;
	      
#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dydap[xx]);
		  alp += CUDA_BLOCK_DIM;
		  xx += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k;
#pragma unroll 4
		  for(; m < l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dydap[m]);
			  alp++;
			}
		      iap++;
		    } /* m */
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	      alpp += mf1;
	      alp = alpp;
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}


__device__ void __forceinline__ MrqcofCurve23I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit + 1;
  int l, jp, j, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  __shared__ double dyda[N80];
  
  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg[1][bid] += lpoints;
    }

  lnp2 = npg[2][bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;

#pragma unroll 
  for(jp = 0; jp < lpoints; jp++) 
    {
      lnp2++;
      //double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&(ytemp[jp]));
      
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l] = a;
	  dyda[l + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
#pragma unroll 1
      while(l < ma)
	{
	  dyda[l] = __ldca(c);
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      //j = 0;
      sig2i = __ldg(&CUDA_sigr2[lnp2]); //__drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      double sig2iwght = sig2i * wght;
      double *betap = beta;
      double * __restrict__ alp = alpha + mf1 + threadIdx.x + 1;
      double *alpp = alp;
#pragma unroll 4
      for(l = 1; l <= lastone; l++)
	{
	  wt = dyda[l] * sig2iwght;
	  int xx = threadIdx.x;
#pragma unroll 2
	  while(xx < l)
	    {
	      __stwb(alp, __ldca(alp) + wt * dyda[xx]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */
	  ++betap;
	  alpp += mf1;
	  if(threadIdx.x == 0)
	    {
	      __stwb(betap, __ldca(betap) + dy * wt);
	    }
	  alp = alpp;
	} /* l */
      
      int * __restrict__ iapp = CUDA_ia;
      alp = alpp;

#pragma unroll 4
      for(; l < lma; l++)
	{
	  ++betap;
	  if(iapp[l + 1])
	    {
	      int xx = threadIdx.x;
	      wt = dyda[l - 1] * sig2iwght;

#pragma unroll 2
	      while(xx < lastone)
		{
		  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} /* m */

	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone;
		  int * __restrict__ iap = iapp + m + 1;
		  alp = alpp + k;
#pragma unroll 4
		  for(; m < l; m++)
		    {
		      if(*iap)
			{
			  __stwb(alp, __ldca(alp) + wt * dyda[xx]);
			  alp++;
			}
		      iap++;
		    } /* m */
		  __stwb(betap, __ldca(betap) + dy * wt);
		}
	    }
	  alpp += mf1;
	  alp = alpp;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg[2][bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}
// curve2 ends



__global__ void CudaCalculatePrepare(int n_start, int n_max)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = n_start + tid;

  if(n > n_max)
    {
      setFlag(isInvalid, tid);
      return;
    }
  else
    {
      resetFlag(isInvalid, tid);
    }

  per_best[tid] = 0; 
  dark_best[tid] = 0;
  la_best[tid] = 0;
  be_best[tid] = 0;
  dev_best[tid] = 1e40;
}


__global__ void 
__launch_bounds__(1024,1)
  CudaCalculatePreparePole(double freq_start, double freq_step, int n_start, double beta, double lambda)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  n_start += tid;

  if(isAnyTrue(isInvalid, tid)) 
    {
      atomicAdd(&CUDA_End, 1);
      isReported[tid] = 0; //signal not to read result

      return;
    }

  double period = __drcp_rn(freq_start - (n_start - 1) * freq_step);
  double * __restrict__ cgp = cgg[tid] + 1; 
  double const * __restrict__ cfp = CUDA_cg_first; // + 1;
  /* starts from the initial ellipsoid */
  int i;
  int ncoef = CUDA_Ncoef;
#pragma unroll 1
  for(i = 1; i <= ncoef - (UNRL - 1); i += UNRL)
    {
      double d[UNRL];
      int ii;
      for(ii = 0; ii < UNRL; ii++)
	d[ii] = *cfp++;
      for(ii = 0; ii < UNRL; ii++)
	*cgp++ = d[ii];
    }
#pragma unroll 3
  for( ; i <= ncoef; i++)
    {
      *cgp++ = *cfp++; 
    }

  
  /* The formulae use beta measured from the pole */
  /* conversion of lambda, beta to radians */
  *cgp++ = DEG2RAD * 90 - DEG2RAD * beta; //CUDA_beta_pole[m];
  *cgp++ = DEG2RAD * lambda; //CUDA_lambda_pole[m];
   
  /* Use omega instead of period */
  *cgp++ = (24.0 * 2.0 * PI) / period; // ****

#pragma unroll
  for(i = 1; i <= CUDA_Nphpar; i++)
    {
      *cgp++ = CUDA_par[i];
      //if(i == 2)
      //  printf("%lf, ", CUDA_par[2]); // 0.1
    }
  
  /* Use logarithmic formulation for Lambert to keep it positive */
  *cgp++ = CUDA_lcl; //log(CUDA_cl); 
  /* Lommel-Seeliger part */
  *cgp++ = 1;

  /* Levenberg-Marquardt loop */
  // moved to global iter_max,iter_min,iter_dif_max
  //
  rchisqg[tid] = -1;
  Alamda[tid] = -1;
  Niter[tid] = 0;
  iter_diffg[tid] = 1e40;
  dev_oldg[tid] = 1e30;
  dev_newg[tid] = 0;
  isReported[tid] = 0;
}


__global__ void CudaCalculateIter1Begin(int n_max)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(tid > n_max) return;

  if(isAnyTrue(isInvalid, tid))
    {
      return;
    }

  int niter = __ldg(&Niter[tid]);
  bool b_isniter = ((niter < CUDA_n_iter_max) && (__ldg(&iter_diffg[tid]) > CUDA_iter_diff_max)) || (niter < CUDA_n_iter_min);

  if(b_isniter)
    setFlag(isNiter, tid);
  else
    resetFlag(isNiter, tid);
  
  if(b_isniter)
    {
      if(__ldg(&Alamda[tid]) < 0)
	{
	  setFlag(isAlambda, tid);

	  Alamda[tid] = CUDA_Alamda_start; /* initial alambda */
	}
      else
	{
	  resetFlag(isAlambda, tid);
	}
    }
  else
    {
      if(!(__ldg(&isReported[tid])))
	{
	  atomicAdd(&CUDA_End, 1);
	  isReported[tid] = 1;
	}
    }

}

__global__ void 
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof1Start(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!(flags & isInvalid)) &&
     (flags & isNiter) &&
     (flags & isAlambda))
    {
      if(threadIdx.x == 0)
	{
	  trial_chisqg[bid] = 0;
	  npg[0][bid] = 0;
	  npg[1][bid] = 0;
	  npg[2][bid] = 0;
	  aveg[bid] = 0;
	}
      mrqcof_start(CUDA_LCC, cgg[bid], alphag[bid] - 1, betag[bid] - 1, bid);
    }
  
  int tid = blockIdx() * blockDim.x + threadIdx.x;
  
  if(tid < blockDim.y * gridDim.x)
    {
      double *a = cgg[tid]; 
      blmatrix(a[CUDA_ma-4-CUDA_Nphpar], a[CUDA_ma-3-CUDA_Nphpar], tid);
    }
}



//XXXXXX 21%
__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(512, 1) //768
#endif  
CudaCalculateIter1Mrqmin1End(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;
  
  mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, /*CUDA_mfit + 1,*/ CUDA_BLOCK_DIM);
}


__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(1024, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqmin2End(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];
  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  mrqmin_2_end(CUDA_LCC, CUDA_ma, bid);

  __syncwarp();
  if(threadIdx.x == 0)
    Niter[bid]++;
}


__global__ void
__launch_bounds__(512, 1) 
CudaCalculateIter1Mrqcof1Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  MrqcofCurve23I0IA0(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, bid);
}


__global__ void
__launch_bounds__(768, 1) 
CudaCalculateIter1Mrqcof1Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  MrqcofCurve23I0IA1(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, bid);
}


__global__ void
__launch_bounds__(512, 1) 
CudaCalculateIter1Mrqcof1Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  MrqcofCurve23I1IA0(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, bid);
}



__global__ void
__launch_bounds__(512, 1) 
CudaCalculateIter1Mrqcof1Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  MrqcofCurve23I1IA1(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, bid);
}




__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  MrqcofCurve23I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  MrqcofCurve23I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}


// SLOW
__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  MrqcofCurve23I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  MrqcofCurve23I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof1CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  double *cg = cgg[bid]; //CUDA_LCC->cg;
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, lpoints, bid);
}


__global__ void 
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof1CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  double *cg = cgg[bid]; //CUDA_LCC->cg;
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, lpoints, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(256, 1) //768
#else
__launch_bounds__(512, 1) //512
#endif  
CudaCalculateIter1Mrqcof1CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  double *cg = cgg[bid]; //CUDA_LCC->cg;
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, lpoints, bid);
}


__global__ void 
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(256, 1) //768
#else
__launch_bounds__(512, 1) //512
#endif  
CudaCalculateIter1Mrqcof1CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  double *cg = cgg[bid]; //CUDA_LCC->cg;
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, alphag[bid] - 1, betag[bid] - 1, lpoints, bid);
}


__global__ 
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
void CudaCalculateIter1Mrqcof1Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  if(CUDA_LCC->ytemp == NULL) return;

  mrqcof_curve1_lastI0(CUDA_LCC, bid);
}


__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof1Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  mrqcof_curve1_lastI1(CUDA_LCC, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof1End(void)
{
  int tid = blockIdx.x * blockDim.y + threadIdx.y;
  auto CUDA_LCC = &CUDA_CC[tid];

  unsigned int flags = getFlags(tid);
  if((!!(flags & isInvalid)) | !(flags & isNiter) | !(flags & isAlambda)) return;

  mrqcof_end(CUDA_LCC, alphag[tid] - 1);
  Ochisq[tid] = trial_chisqg[tid];
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Start(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];
  int tid = blockIdx() * blockDim.x + threadIdx.x;

  unsigned int flags = getFlags(bid);
  if((!(flags & isInvalid)) &&
     (flags & isNiter))
    {
      if(threadIdx.x == 0)
	{
	  trial_chisqg[bid] = 0;
	  npg[0][bid] = 0;
	  npg[1][bid] = 0;
	  npg[2][bid] = 0;
	  aveg[bid] = 0;
	}
        
      mrqcof_start(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
    }

  if(tid < blockDim.y * gridDim.x)
    {
      double *a = atry[tid]; 
      blmatrix(a[CUDA_ma - CUDA_Nphpar - 4], a[CUDA_ma - CUDA_Nphpar - 3], tid);
    }

}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  double *atryp = atry[bid];
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(384, 1) //768
#else
__launch_bounds__(768, 1) //768
#endif  
CudaCalculateIter1Mrqcof2CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  double *atryp = atry[bid]; 
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(256, 1) //768
#else
__launch_bounds__(512, 1) //512
#endif  
CudaCalculateIter1Mrqcof2CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  double *atryp = atry[bid];
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

//ZZZ
/* MOST TIME CONSUMINNG KERNEL MRQCOF2CURVEM12I1IA0*/

__global__ void 
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(256, 1) //768
#else
__launch_bounds__(448, 1) //512,, 448 fast
#endif  
CudaCalculateIter1Mrqcof2CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  double *atryp = atry[bid]; 
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

//ZZZ
__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  mrqcof_curve1_lastI0(CUDA_LCC, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof2Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  mrqcof_curve1_lastI1(CUDA_LCC, bid);
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter1Mrqcof2End(void)
{
  int tid = blockIdx.x * blockDim.y + threadIdx.y;
  auto CUDA_LCC = &CUDA_CC[tid];

  unsigned int flags = getFlags(tid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  mrqcof_end(CUDA_LCC, CUDA_LCC->covar);
  Chisq[tid] = __ldg(&trial_chisqg[tid]);
}


__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(1024, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateFinishPole(void)
{
  int bid = blockIdx();

  unsigned int flags = getFlags(bid);
  if(!!(flags & isInvalid)) return;

  double dn = __ldca(&dev_newg[bid]), db = __ldca(&dev_best[bid]);
  int nf = CUDA_Numfac;

  if(dn >= db)
    return;

  double tot = 0, tot2 = 0;
  int xx = threadIdx.x;
  double const * __restrict__ p = &Areag[bid][xx]; 
#pragma unroll 1
  for( ; xx < nf - (CUDA_BLOCK_DIM * 4 - 1); xx += 4 * CUDA_BLOCK_DIM)
    {
      double a[4];
#pragma unroll      
      for(int i = 0; i < 4; i++)
	a[i] = p[i * CUDA_BLOCK_DIM];
      a[0] += a[1];
      a[2] += a[3];
      tot += a[0];
      tot2 += a[2];
      p += CUDA_BLOCK_DIM * 4;
    }
  tot += tot2;
#pragma unroll 2
  for( ; xx < nf; xx += CUDA_BLOCK_DIM)
    {
      tot +=  *p;
      p += CUDA_BLOCK_DIM;
    }
  tot += __shfl_down_sync(0xffffffff, tot, 16);
  tot += __shfl_down_sync(0xffffffff, tot, 8);
  tot += __shfl_down_sync(0xffffffff, tot, 4);
  tot += __shfl_down_sync(0xffffffff, tot, 2);
  tot += __shfl_down_sync(0xffffffff, tot, 1);
  if(threadIdx.x == 0)
    {
      tot = __drcp_rn(tot);
      
      double dark = __ldca(&chck[bid]); 
      /* period solution */
      double *cggp = cgg[bid];
      double dd = dark * 100.0 * tot;
      if(isnan(dd) == 1)
	dd = 1.0;
      double period = 2 * PI / __ldca(&cggp[CUDA_Ncoef + 3]);
      
      /* pole solution */
      double la_tmp = RAD2DEG * __ldca(&cggp[CUDA_Ncoef + 2]);
      double be_tmp = 90 - RAD2DEG * __ldca(&cggp[CUDA_Ncoef + 1]);
      
      dev_best[bid]  = dn;
      dark_best[bid] = dd;
      per_best[bid]  = period;
      la_best[bid]   = la_tmp + (la_tmp < 0 ? 360 : 0);
      be_best[bid]   = be_tmp;
    }
}



__global__ void
#if (__CUDA_ARCH__ < 700)
__launch_bounds__(512, 1) //768
#else
__launch_bounds__(1024, 1) //768
#endif  
CudaCalculateIter2(void)
{
  int bid = blockIdx();
  unsigned int flags = getFlags(bid);
  if((!!(flags & isInvalid)) | !(flags & isNiter)) return;

  int nf = CUDA_Numfac;

  auto CUDA_LCC = &CUDA_CC[bid];

  double chisq = __ldg(&Chisq[bid]);
  double ochisq = __ldg(&Ochisq[bid]);

  if(Niter[bid] == 1 || chisq < ochisq)
    {
      curv(CUDA_LCC, cgg[bid], bid);
      
      double a[3] = {0, 0, 0};

      int j = threadIdx.x;

      double const * __restrict__ areap = Areag[bid];

#pragma unroll 2
      while(j < nf - 3*CUDA_BLOCK_DIM)
	{
	  double dd0 = areap[j]; // __ldca
	  double dd1 = areap[j + CUDA_BLOCK_DIM];
	  double dd2 = areap[j + 2*CUDA_BLOCK_DIM];
	  double dd3 = areap[j + 3*CUDA_BLOCK_DIM];
#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    {
	      a[i] +=
		dd0 * CUDA_Nor[i][j] +
		dd1 * CUDA_Nor[i][j + CUDA_BLOCK_DIM] +
		dd2 * CUDA_Nor[i][j + 2*CUDA_BLOCK_DIM] +
		dd3 * CUDA_Nor[i][j + 3*CUDA_BLOCK_DIM];
	    }
	  j += 4*CUDA_BLOCK_DIM;
	}
      //#pragma unroll 2
      if(j < nf - CUDA_BLOCK_DIM)
	{
	  double dd0 = areap[j]; // __ldca
	  double dd1 = areap[j + CUDA_BLOCK_DIM];
#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    {
	      a[i] +=
		dd0 * CUDA_Nor[i][j] +
		dd1 * CUDA_Nor[i][j + CUDA_BLOCK_DIM];
	    }
	  j += 2*CUDA_BLOCK_DIM;
	}
      if(j < nf) //while
	{
	  double dd = areap[j];
#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    {
	      //double const * __restrict__ norp = CUDA_Nor[i];
	      a[i] += dd * CUDA_Nor[i][j]; //__ldca(&norp[j]);
	    }
	  j += CUDA_BLOCK_DIM;
	}
      
#pragma unroll
      for(int off = CUDA_BLOCK_DIM/2; off > 0; off >>= 1)
	{
	  double b[3];
#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    b[i] = __shfl_down_sync(0xffffffff, a[i], off);
#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    a[i] += b[i];
	}
      
      //__syncwarp();
      if(threadIdx.x == 0)
	{
	  double conwr2 = CUDA_conw_r, aa = 0;
	  
	  Ochisq[bid] = chisq;
	  conwr2 *= conwr2;

#pragma unroll 
	  for(int i = 0; i < 3; i++)
	    {
	      aa += a[i]*a[i];
	    }
	  
	  double rchisq = chisq - aa * conwr2;
	  double dev_old = dev_oldg[bid];
	  double dev_new = __dsqrt_rn(rchisq / (CUDA_ndata - 3));
	  chck[bid] = norm3d(a[0], a[1], a[2]);

	  dev_newg[bid]  = dev_new;
	  double diff    = dev_old - dev_new;
	  
	  /* 
	  // only if this step is better than the previous,
	  // 1e-10 is for numeric errors 
	  */
	  
	  if(diff > 1e-10)
	    {
	      iter_diffg[bid] = diff; 
	      dev_oldg[bid] = dev_new; 
	    }
	}
    }
}


__global__ void CudaCalculateFinish(void) //  not used
{
}



__global__ void test(float *p)
{
}
