//#pragma OPENCL EXTENSION cl_khr_printf : enable
__kernel void curv(const __global float *cg, __global double *Area, const __global double *Darea, 
    const __global double *Fc, const int maxFcX, const int maxFcY,
    const __global double *Fs, const int maxFsX, const int maxFsY,
    const __global double *Pleg, const int maxPlegX, const int maxPlegY, const int maxPlegZ,
    const int Mmax, const int Lmax,
    __global double *Dg, const int maxDgX, const int maxDgY,
    const __global double *Dsph, const int maxDsphX, const int msxDsphY)
{

    int i, m, l, k, n = 0;
    double g = 0;

    i = get_global_id(0) + 1;
    

    for (l = 0; l <= Lmax; l++)
    {
        double fsum;
        n++;
        fsum = cg[n] * Fc[i + maxFcX*0];
        g = g + Pleg[i + maxPlegX*l + maxPlegY*0] * fsum;
    }
    for (m = 1; m <= Mmax; m++) {
        for (l = m; l <= m; l++) {
            double fsum;
            n++;
            fsum = cg[n] * Fc[i + maxFcX*m];
            n++;
            fsum = fsum + cg[n] * Fs[i + maxFcX*m];
            g = g + Pleg[i + maxPlegX*l + maxPlegY*m] * fsum;
        }
    }
    g = exp(g);
    Area[i - 1] = Darea[i - 1] * g;

    //for (k = 1; k <= n; k++)
    //{
    //    Dg[i - 1 + maxDgX * (k - 1)] = g * Dsph[i + maxDsphX * k]
    //    //Dg[i - 1][k - 1] = g * Dsph[i][k];
    //}
}