/* rotation matrix and its derivatives 
   converted from Mikko's fortran code

   8.11.2006
*/

#include <math.h>
#include "globals_CUDA.h"

__device__ void matrix(freq_context *CUDA_LCC,double omg, double t, double tmat[][4], double dtm[][4][4])
{
   double f, cf, sf, dfm[4][4], fmat[4][4]; 
   
   int i, j, k;
   
   /* phase of rotation */
   f = omg * t + CUDA_Phi_0;
   f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
   cf = cos(f);
   sf = sin(f);
   /* rotation matrix, Z axis, angle f */ 
   fmat[1][1] = cf;
   fmat[1][2] = sf;
   fmat[1][3] = 0;
   fmat[2][1] = -sf;
   fmat[2][2] = cf;
   fmat[2][3] = 0;
   fmat[3][1] = 0;
   fmat[3][2] = 0;
   fmat[3][3] = 1;
   /* Ders. w.r.t omg */
   dfm[1][1] = -t * sf;
   dfm[1][2] = t * cf;
   dfm[1][3] = 0;
   dfm[2][1] = -t * cf;
   dfm[2][2] = -t * sf;
   dfm[2][3] = 0;
   dfm[3][1] = 0;
   dfm[3][2] = 0;
   dfm[3][3] = 0;
   /* Construct tmat (complete rotation matrix) and its derivatives */
   for (i = 1; i <= 3; i++)
      for (j = 1; j <= 3; j++)
      {
         tmat[i][j] = 0;
         dtm[1][i][j] = 0;
         dtm[2][i][j] = 0;
         dtm[3][i][j] = 0;
	 for (k = 1; k <= 3; k++)
	 {
            tmat[i][j] = tmat[i][j] + fmat[i][k] * (*CUDA_LCC).Blmat[k][j];
            dtm[1][i][j] = dtm[1][i][j] + fmat[i][k] * (*CUDA_LCC).Dblm[1][k][j];
            dtm[2][i][j] = dtm[2][i][j] + fmat[i][k] * (*CUDA_LCC).Dblm[2][k][j];
            dtm[3][i][j] = dtm[3][i][j] + dfm[i][k] * (*CUDA_LCC).Blmat[k][j];
          }
      }
}
