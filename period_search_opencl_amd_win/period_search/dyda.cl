__kernel void dyda(__global double* dbr, 
    __global double* dg,
    __global double* dyda,
    __global int* incl,
    unsigned int col,
    unsigned int numRows)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int s = get_global_size(0);
    size_t id = (j * s + i);
    /*if (i == 0 && j == 0)
    {
        printf("dg[0]: %.9f\tsbr[0]: %.9f\n", dg[i], dbr[i]);
    }*/

    //printf("gid[0]=%d\tgid[1]=%d\tgsize[0]=%d\n", i, j, s);

    if (i >= numRows || j >= col) return;
    //printf("i:%d, j:%d\n", i, j);

    unsigned int incl_id = incl[i];
    unsigned int y = incl_id * 256 + j;
    dyda[id] = dbr[i] * dg[y];
    //printf("dg[%d][%d]:\t%f", incl_id, col, dg[y]);
}