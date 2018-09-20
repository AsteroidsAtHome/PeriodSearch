/* Find the curv. fn. Laplace series for given ellipsoid
   converted from Mikko's fortran code

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "declarations.hpp"
#include "AnglesOfNormals.hpp"

void ellfit(double cg[], double a, double b, double c, int ncoef, struct AnglesOfNormals angles_n)
{
   int i, m, l, n, j, k;
   int *indx;

   double sum, st;
   double *fitvec, *d, *er,
          **fmat, **fitmat;

   indx = vector_int(ncoef);

   fitvec = vector_double(ncoef);
   er = vector_double(angles_n.number_facets);
   d = vector_double(1);

   fmat = matrix_double(angles_n.number_facets, ncoef);
   fitmat = matrix_double(ncoef, ncoef);

   /* Compute the LOGcurv.func. of the ellipsoid */
   for (i = 1; i <= angles_n.number_facets; i++)
   {
      st = sin(angles_n.theta_angle.at(i));
      sum = pow(a * st * cos(angles_n.phi_angle.at(i)), 2) + pow(b * st * sin(angles_n.phi_angle.at(i)), 2) +
            pow(c * cos(angles_n.theta_angle.at(i)), 2);
      er[i] = 2 * (log(a * b * c) - log(sum));
   }
   /* Compute the sph. harm. values at each direction and
      construct the matrix fmat from them */
   for (i = 1; i <= angles_n.number_facets; i++)
   {
      n = 0;
      for (m = 0; m <= Mmax; m++)
         for (l = m; l <= Lmax; l++)
     {
            n++;
            if (m != 0)
        {
               fmat[i][n] = Pleg[i][l][m] * cos(m * angles_n.phi_angle.at(i));
               n++;
               fmat[i][n] = Pleg[i][l][m] * sin(m * angles_n.phi_angle.at(i));
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

         for (k = 1; k <= angles_n.number_facets; k++)
            fitmat[i][j] = fitmat[i][j] + fmat[k][i] * fmat[k][j];

      }
      fitvec[i]=0;

      for (j = 1; j <= angles_n.number_facets; j++)
         fitvec[i] = fitvec[i] + fmat[j][i] * er[j];
   }


   ludcmp(fitmat,ncoef,indx,d);
   lubksb(fitmat,ncoef,indx,fitvec);

   for (i = 1; i <= ncoef; i++)
      cg[i] = fitvec[i];

   deallocate_matrix_double(fitmat, ncoef);
   deallocate_matrix_double(fmat, angles_n.number_facets);
   deallocate_vector((void *) fitvec);
   deallocate_vector((void *) d);
   deallocate_vector((void *) indx);
   deallocate_vector((void *) er);

}

