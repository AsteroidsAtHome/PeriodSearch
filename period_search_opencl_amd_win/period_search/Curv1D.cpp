#include "globals.h"
#include "Array2D.cpp"
#include "Array3D.cpp"
#include "OpenClWorker.hpp"

void curv1D(double cg[])
{
    cl_double *_cg = cg;                                                    // write
    cl_double *_area = Area;                                                // write
    cl_double *_darea = Darea;                                              // read
    Array2D<cl_double, MAX_N_FAC + 1, MAX_LM + 1> _fc(Fc);                  // read
    Array2D<cl_double, MAX_N_FAC + 1, MAX_LM + 1> _fs(Fs);                  // read
    Array2D<cl_double, MAX_N_FAC + 1, MAX_N_PAR + 1> _dg(Dg);               // write
    Array2D<cl_double, MAX_N_FAC + 1, MAX_N_PAR + 1> _dsph(Dsph);           // read
    Array3D<cl_double, MAX_N_FAC + 1, MAX_LM + 1, MAX_LM + 1> _pleg(Pleg);  // read

    //curvCl();

    for (int i = 1; i <= Numfac; i++)
    {
        cl_double g = 0;
        int n = 0;
        for (int m = 0; m <= Mmax; m++)
            for (int l = m; l <= Lmax; l++)
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
        }
    }
}
