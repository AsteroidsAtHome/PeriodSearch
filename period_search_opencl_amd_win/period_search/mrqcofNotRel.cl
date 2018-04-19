#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void mrqcofNotRel(ulong iter, ulong ma, ulong dytempCol, ulong alphaCol,
    const __global double *ytemp,
    const __global double *dytemp,
    const __global double *sig,
    const __global double *y,
    const __global double *weight,
    __global double *_sig2iwght,
    __global double *_dy,
    __global double *_alpha,
    __global double *_beta)
{

    uint jp = get_global_id(0);
    if (jp > iter) {
        _sig2iwght[jp] = (double)0.0;
        _dy[jp] = (double)0.0;

        return;
    }
    _sig2iwght[jp] = weight[jp] * (1 / (sig[jp] * sig[jp]));
    if (jp <= 10) {
        printf("kernel: y[%d]:\t%.9f\n", jp, y[jp]);
    }
    _dy[jp] = y[jp] - ytemp[jp];
    
    double wt = dytemp[jp * dytempCol + 1] * _sig2iwght[jp];
    _alpha[0] = _alpha[0] + wt * dytemp[jp * dytempCol + 1];
    _beta[0] = _beta[0] + _dy[jp] * wt;
}