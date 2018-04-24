//#pragma OPENCL EXTENSION cl_khr_printf : enable
kernel void curv(__global const double cg[],
                __global const double fc[],
                __global const double fs[],
                __global const double dsph[],
                __global const double pleg[],
                __global const double darea[],
                __global double dg[],
                __global double area[],
    uint Lmax, uint Mmax, uint yFc, uint, yFs, uint yPleg, uint zPleg)
{
    int i, l;
    i = get_global_id(0);
    if (i == 0) return;
    for (l = 0; l < 10; l++) 
    {
        //
    }
    double g = 0.0;
    uint n = 0;
    for(int m = 0; m <= Mmax; m++)
        for (int l = m; l <= Lmax; l++)
        {
            n++;
            double fsum = cg[n] * fc[i * yFc + m];
            if (m > 0)
            {
                n++;
                fsum = fsum + cg[n] * fs[i * yFs + m];
            }
            g = g + pleg[i * zPleg*yPleg + l * zPleg + m];
        }

    g = exp(g);
    area[i - 1] = darea[i - 1] * g;

    /*cl_double g = 0;
    int n = 0;
    for (int m = 0; m <= Mmax; m++)
        for (l = m; l <= Lmax; l++)
        {
            n++;
            cl_double fsum = cg[n] * _fc(i, m);
            if (m > 0) {
                n++;
                fsum = fsum + _cg[n] * _fs(i, m);
            }
            g = g + _pleg(i, l, m) * fsum;
        }

    g = exp(g);
    _area[i - 1] = _darea[i - 1] * g;

    for (int k = 1; k <= n; k++)
    {
        _dg.set(i - 1, k - 1, g * _dsph(i, k));
    }*/
   
}