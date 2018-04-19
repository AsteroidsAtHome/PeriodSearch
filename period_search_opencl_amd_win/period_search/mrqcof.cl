#pragma OPENCL EXTENSION cl_amd_printf : enable
kernel void mrqcof(ulong lpoints, ulong ma, double ave, 
    __global double sig[], __global double dave[],
    __global double ytemp[], __global double dytemp[],
    ulong col) {
    // 'daytemp[]' is of type aligned Array2D

    int jp = get_global_id(0);
    printf("global_id: %d\n", jp);
    if (jp <= lpoints)
    {
        double coef = sig[jp] * lpoints / ave;
        for (int l = 1; l <= ma; l++)
        {
            //dytemp[jp][l] = coef * (dytemp[jp][l] - ytemp[jp] * dave[l] / ave);
            dytemp[jp * col + l] = coef * (dytemp[jp * col + l] - ytemp[jp] * dave[l] / ave);
        }
        ytemp[jp] = coef * ytemp[jp];
        //dytemp[jp][1] = 0;  // Set the size scale 1coeff. deriv. explicitly zero for relative lcurves 
        dytemp[jp * col + 1] = 0;
    }
}