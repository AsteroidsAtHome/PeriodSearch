#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void mrqcofMid(__global uint *res, ulong lpoints) {

    uint j = get_global_id(0);
    if (j >= lpoints) return;
    //printf("global_id: %d\tlpoints: %d\n ", j, lpoints);
    res[j] = res[j] +j;
}