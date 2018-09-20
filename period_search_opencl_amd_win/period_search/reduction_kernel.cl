__kernel void reduce2(__global double* g_idata, __global double* g_odata, __global double* dbr,
    __global double* dg,
    __global int* incl,
    unsigned int col,
    unsigned int n, 
    __local double* sdata,
    unsigned int numRows)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    if (i >= numRows) return;


    sdata[tid] = (i < n) ? g_idata[i] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    /*unsigned int y = incl_id * 256 + col;
    print("dg[%d][%d]: %f\n", incl_id, col, dg[y]);
    return;*/

    // do reduction in shared mem
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            int incl_id = incl[i];
            //printf("incl: %d\t256\t%d\t", incl_id, col);
            unsigned int y = incl_id * 256 + col;  // send '256' as 'numCols' parameter 
            //printf("dg[%d]: %f\n", y, dg[y]);
            unsigned int second = i + s;
            if (second < numRows)
            {
                sdata[tid] += (sdata[tid + s] + dbr[i + s] * dg[y]);
            }
            else
            {
                sdata[tid] = sdata[tid] + dbr[tid] * dg[y];
            }

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}