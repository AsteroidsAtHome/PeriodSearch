__kernel void dave(__global double *dave, __global const double *dyda) {
    uint l = get_global_id(0) + 1;
    dave[l] = dave[l] + dyda[l - 1];
}