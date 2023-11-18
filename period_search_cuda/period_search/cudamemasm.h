/*
** petri33:
**
** Cuda 7.5 compiler produces 2 float2 store operations from *(float4*)p = f4;
** These helper functions implement st.128 in one operation.
**
** Other helper for cache configuration on load/store operations
*/

// store 1 float caching (likely to be reused soon)
__device__ void inline ST_f_wb(float *addr, float x)
{
  asm("st.global.wb.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 int32 streaming, not caching (not likely to be reused soon)
__device__ void inline ST_i_cs(int *addr, int x)
{
  asm("st.global.cs.s32 [%0], %1;" :: "l"(addr) ,"r"(x));
}

// store 1 int32 caching (likely to be reused soon)
__device__ void inline ST_i_wb(int *addr, int x)
{
  asm("st.global.wb.s32 [%0], %1;" :: "l"(addr) ,"r"(x));
}

// store 1 int32 caching (likely to be reused)
__device__ void inline ST_i_cg(int *addr, int x)
{
  asm("st.global.cg.s32 [%0], %1;" :: "l"(addr) ,"r"(x));
}

//nocache
__device__ void inline ST_i_wt(int *addr, int x)
{
  asm("st.global.wt.s32 [%0], %1;" :: "l"(addr) ,"r"(x));
}

// store 1 float  not caching (not likely to be reused soon)
__device__ void inline ST_f_wt(float *addr, float x)
{
  asm("st.global.wt.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 float streaming, not caching (not likely to be reused soon)
__device__ void inline ST_f_cs(float *addr, float x)
{
  asm("st.global.cs.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 1 float caching L2 (likely to be reused)
__device__ void inline ST_f_cg(float *addr, float x)
{
  asm("st.global.cg.f32 [%0], %1;" :: "l"(addr) ,"f"(x));
}

// store 2 floats caching L2 (likely to be reused)
__device__ void inline ST_f2_cg(float2 *addr, float2 val)
{
  asm("st.global.cg.v2.f32 [%0], {%1,%2};" :: "l"(addr) ,"f"(val.x),"f"(val.y));
}

// store 4 floats caching L2 (likely to be reused)
__device__ void inline ST_f4_cg(float4 *addr, float4 val)
{
  asm("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(val.x),"f"(val.y),"f"(val.z),"f"(val.w));
}

// store 4 floats
__device__ void inline ST_4f(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4
__device__ void inline ST_f4(float4 *addr, float4 val)
{
  ST_4f(addr, val.x, val.y, val.z, val.w);
}

// store 2 floats caching (likely to be reused soon)
__device__ void inline ST_2f_wb(float2 *addr, float x, float y)
{
  asm("st.global.wb.v2.f32 [%0], {%1,%2};" :: "l"(addr) ,"f"(x),"f"(y));
}

// store float2 (likely to be reused soon)
__device__ void inline ST_f2_wb(float2 *addr, float2 val)
{
  ST_2f_wb(addr, val.x, val.y);
}

// store 4 floats caching (likely to be reused soon)
__device__ void inline ST_4f_wb(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.wb.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4 (likely to be reused soon)
__device__ void inline ST_f4_wb(float4 *addr, float4 val)
{
  ST_4f_wb(addr, val.x, val.y, val.z, val.w);
}

// store 4 floats non caching 
__device__ void inline ST_f4_wt(float4 *addr, float4 val)
{
  asm("st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(val.x),"f"(val.y),"f"(val.z),"f"(val.w));
}

// store 2 floats streaming (not likely to be reused immediately)
__device__ void inline ST_2f_cs(float2 *addr, float x, float y)
{
  asm("st.global.cs.v2.f32 [%0], {%1,%2};" :: "l"(addr) ,"f"(x),"f"(y));
}

// store float4 (no likely to be reused immediately)
__device__ void inline ST_f2_cs(float2 *addr, float2 val)
{
  ST_2f_cs(addr, val.x, val.y);
}

// store 4 floats streaming (not likely to be reused immediately)
__device__ void inline ST_4f_cs(float4 *addr, float x, float y, float z, float w)
{
  asm("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr) ,"f"(x),"f"(y),"f"(z),"f"(w));
}

// store float4 (no likely to be reused immediately)
__device__ void inline ST_f4_cs(float4 *addr, float4 val)
{
  ST_4f_cs(addr, val.x, val.y, val.z, val.w);
}

//load through nonuniform cache

//ca = L1,L2
#if (__CUDA_ARCH__ > 350)
__device__ float inline LDG_f_ca(float *addr, const int offset)
{
  float v;
  asm("ld.global.ca.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr + offset));
  return v; 
}

__device__ double inline LDG_d_ca(double *addr, const int offset)
{
  double v;
  asm("ld.global.ca.nc.f64 %0, [%1];" : "=d"(v) : "l"(addr + offset));
  return v; 
}

#else
__device__ float inline LDG_f_ca(float *addr, const int offset) //GTX780
{
  float v;
  asm("ld.global.ca.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr + offset));
  return v; 
}

__device__ double inline LDG_d_ca(double *addr, const int offset)
{
  double v;
  asm("ld.global.ca.nc.f64 %0, [%1];" : "=d"(v) : "l"(addr + offset));
  return v; 
}
#endif

__device__ float inline LDG_f_nc(float *addr, const int offset)
{
  float v;
  asm("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr + offset));
  return v; 
}

__device__ float2 inline LDG_f2_ca(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.ca.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float2 inline LDG_f2_nc(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LDG_f4_ca(float4 *addr, const int offset)
{
  float4 v;

  asm("ld.global.ca.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LDG_f4_nc(float4 *addr, const int offset)
{
  float4 v;

  asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//cg = L2
__device__ float inline LDG_f_cg(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm ("ld.global.cg.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_cg(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.cg.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LDG_f4_cg(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.cg.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//streaming once through L1,L2
__device__ float inline LDG_f_cs(float *addr, const int offset)
{
  float v;
  addr += offset;
  asm volatile ("ld.global.cs.nc.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v; 
}

__device__ float2 inline LDG_f2_cs(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.cs.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LDG_f4_cs(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.cs.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//last use L1,L2 (same as cs on global addresses)
__device__ float inline LDG_f_lu(float *addr, const int offset)
{
  float v;
  asm("ld.global.lu.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}

__device__ float2 inline LDG_f2_lu(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.lu.nc.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LDG_f4_lu(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.lu.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

// bypass cache
__device__ float4 inline LDG_f4_cv(float4 *addr, const int offset) //volatile
{
  float4 v;
  asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

__device__ float inline LDG_f_cv(float *addr, const int offset) //volatile
{
  float v;
  asm("ld.global.cv.f32 {%0}, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}


// load through uniform cache (faster but smaller cache)

//ca = L1,L2
//#if (__CUDA_ARCH__ > 350)
__device__ float inline LD_f_ca(float *addr, const int offset)
{
  float v;
  asm("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}
//#else
//__device__ float inline LDG_f_ca(float *addr, const int offset)
//{
//  float v;
//  asm("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
//  return v; 
//}
//#endif

__device__ float2 inline LD_f2_ca(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.ca.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LD_f4_ca(float4 *addr, const int offset)
{
  float4 v;

  asm("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//cg = L2
__device__ float inline LD_f_cg(float *addr, const int offset)
{
  float v;
  asm("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}

__device__ float2 inline LD_f2_cg(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LD_f4_cg(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//streaming once through L1,L2
__device__ float inline LD_f_cs(float *addr, const int offset)
{
  float v;
  asm("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}

__device__ float2 inline LD_f2_cs(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LD_f4_cs(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}

//last use L1,L2 (same as cs on global addresses)
__device__ float inline LD_f_lu(float *addr, const int offset)
{
  float v;
  asm("ld.global.lu.f32 %0, [%1];" : "=f"(v) : "l"(addr+offset));
  return v; 
}

__device__ float2 inline LD_f2_lu(float2 *addr, const int offset)
{
  float2 v;
  asm("ld.global.lu.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(addr+offset));
  return v; 
}

__device__ float4 inline LD_f4_lu(float4 *addr, const int offset)
{
  float4 v;
  asm("ld.global.lu.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(addr+offset));
  return v; 
}




__device__ void inline prefetch_l1(void *addr)
{
  asm volatile ("prefetch.global.L1 [%0];"::"l"(addr) );
}

__device__ void inline prefetch_l2(void *addr)
{
  asm volatile ("prefetch.global.L2 [%0];"::"l"(addr) );
}



__device__ float inline __fmul_sat(float a, float b)
{
  float res;
  asm("mul.rn.sat.f32 %0, %1, %2 ;" : "=f"(res) : "f"(a), "f"(b));
  return res;
}

__device__ double inline __cvt_d_rni(double a)
{
  double res;
  asm("cvt.rni.f64.f64 %0, %1 ;" : "=d"(res) : "d"(a));
  return res;
}

__device__ float inline __cvt_f_rni(float a)
{
  float res;
  asm("cvt.rni.f32.f32 %0, %1 ;" : "=f"(res) : "f"(a));
  return res;
}

__device__ float inline __cvt_d_to_f(double a)
{
  float res;
  asm("cvt.rz.f32.f64 %0, %1 ;" : "=f"(res) : "d"(a));
  return res;
}

__device__ double inline ___drcp_rn(double a)
{
  double res;
  asm("rcp.approx.ftz.f64 %0, %1 ;" : "=d"(res) : "d"(a));
  return res;
}


__device__ uint inline get_smid(void)
{
  uint ret;
  asm("mov.u32 %0, %%smid ;" : "=r"(ret) );
  return ret; 
}

