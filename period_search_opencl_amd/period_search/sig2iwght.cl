__kernel void sig(__global const double *sig,
    __global const double *weight,
    __global double *sig2iwght,
    __global double *dy, 
    __global const double *y,
    __global const double *ymod,
    const int offset)
{
    double sig2i;
    double wght;
    uint np2 = get_global_id(0) + offset + 1;
    //printf("%d, ", np2);

    sig2i = 1 / (sig[np2] * sig[np2]);
    wght = weight[np2];
    sig2iwght[np2] = sig2i * wght;
    dy[np2] = y[np2] - ymod[np2];
}