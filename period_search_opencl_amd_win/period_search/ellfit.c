/* Find the curv. fn. Laplace series for given ellipsoid
   converted from Mikko's fortran code

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <algorithm>
#include "globals.h"
#include "declarations.hpp"
#include "AnglesOfNormals.hpp"
#include "globals.hpp"
using namespace std;

void ellfit(double cg[], double a, double b, double c,const int& ncoef, const struct AnglesOfNormals& normals)
{
   int i, m, l, n, j;
   int *indx;
   const auto nFacets = static_cast<int>(normals.numberFacets);

   double sum, st;
   double  *fitvec, *er, **fmat, **fitmat; //*d,

   indx = vector_int(ncoef);
   fitvec = vector_double(ncoef);
   er = vector_double(nFacets);
   fmat = matrix_double(nFacets, ncoef);
   fitmat = matrix_double(ncoef, ncoef);
   //d = vector_double(1);

   /* Compute the LOGcurv.func. of the ellipsoid */
   for (i = 1; i <= nFacets; i++)
   {
      st = sin(normals.theta.at(i));
      sum = pow(a * st * cos(normals.phi.at(i)), 2) + pow(b * st * sin(normals.phi.at(i)), 2) +
            pow(c * cos(normals.theta.at(i)), 2);
      er[i] = 2 * (log(a * b * c) - log(sum));
   }
   /* Compute the sph. harm. values at each direction and
      construct the matrix fmat from them */
   for (i = 1; i <= nFacets; i++)
   {
      n = 0;
      for (m = 0; m <= Mmax; m++)
         for (l = m; l <= Lmax; l++)
     {
            n++;
            if (m != 0)
        {
                fmat[i][n] = Pleg[i][l][m] * cos(m * normals.phi.at(i));
               n++;
               fmat[i][n] = Pleg[i][l][m] * sin(m * normals.phi.at(i));
            }
            else
                fmat[i][n] = Pleg[i][l][m];
         }
   }

   /* Fit the coefficients r from fmat[ndir,ncoef]*r[ncoef]=er[ndir] */
   for (i = 1; i <= ncoef; i++)
   {
      for (j = 1; j <= ncoef; j++)
      {
          fitmat[i][j]=0;

         for (auto k = 1; k <= nFacets; k++)
             fitmat[i][j] = fitmat[i][j] + fmat[k][i] * fmat[k][j];

      }
      fitvec[i]=0;

      for (j = 1; j <= nFacets; j++)
          fitvec[i] = fitvec[i] + fmat[j][i] * er[j];
   }

   ludcmp(fitmat,ncoef,indx);
   lubksb(fitmat,ncoef,indx,fitvec);

   for (i = 1; i <= ncoef; i++)
      cg[i] = fitvec[i];

   deallocate_matrix_double(fitmat, ncoef);
   deallocate_matrix_double(fmat, nFacets);
   deallocate_vector((void *) fitvec);
   deallocate_vector((void *) indx);
   deallocate_vector((void *) er);
   //deallocate_vector((void *) d);

}

