class Data {
    double cg;
    //double *Area;
    //double *Darea;
    //double *Fc;
};

//#pragma OPENCL EXTENSION cl_khr_printf : enable
kernel void curv(__global Data *d)
    //const __global float *cg, __global double *Area, const __global double *Darea, 
    //const __global double *Fc, const __global double *Fs, const __global double *Pleg,
    //__global double *Dg, const __global double *Dsph, 
    //const int Mmax, const int Lmax)
{
    int i, l;
    i = get_global_id(0);
    for (l = 0; l < 10; l++) 
    {
        printf("%.9f\n", d->cg[l+i]);
    }
    /*cl_double *_cg = cg;
    cl_double *_area = Area;
    cl_double *_darea = Darea;


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
    Area[i - 1] = Darea[i - 1] * g;*/

    //for (k = 1; k <= n; k++)
    //{
    //    Dg[i - 1 + maxDgX * (k - 1)] = g * Dsph[i + maxDsphX * k]
    //    //Dg[i - 1][k - 1] = g * Dsph[i][k];
    //}
}