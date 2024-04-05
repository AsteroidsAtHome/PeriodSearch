/* beta, lambda rotation matrix and its derivatives 
   
   8.11.2006
*/

#include <math.h>
#include "globals.h"

void blmatrix(double bet, double lam)
{
   double cb, sb, cl, sl;

   cb = cos(bet);
   sb = sin(bet);
   cl = cos(lam);
   sl = sin(lam);
   Blmat[1][1] = cb * cl;
   Blmat[1][2] = cb * sl;
   Blmat[1][3] = -sb;
   Blmat[2][1] = -sl;
   Blmat[2][2] = cl;
   Blmat[2][3] = 0;
   Blmat[3][1] = sb * cl;
   Blmat[3][2] = sb * sl;
   Blmat[3][3] = cb;
   /* Ders. of Blmat w.r.t. bet */
   Dblm[1][1][1] = -sb * cl;
   Dblm[1][1][2] = -sb * sl;
   Dblm[1][1][3] = -cb;
   Dblm[1][2][1] = 0;
   Dblm[1][2][2] = 0;
   Dblm[1][2][3] = 0;
   Dblm[1][3][1] = cb * cl;
   Dblm[1][3][2] = cb * sl;
   Dblm[1][3][3] = -sb;
   /* Ders. w.r.t. lam */
   Dblm[2][1][1] = -cb * sl;
   Dblm[2][1][2] = cb * cl;
   Dblm[2][1][3] = 0;
   Dblm[2][2][1] = -cl;
   Dblm[2][2][2] = -sl;
   Dblm[2][2][3] = 0;
   Dblm[2][3][1] = -sb * sl;
   Dblm[2][3][2] = sb * cl;
   Dblm[2][3][3] = 0;
}
