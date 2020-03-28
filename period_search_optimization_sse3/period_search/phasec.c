/* linear-exponential phase function
   converted from Mikko's fortran code

   8.11.2006
*/

#include <math.h>
#include <stdio.h>
#include "globals.h"

void phasec(double dcdp[], double alpha, double p[])
{
   double e, c;

   /* Exp-lin model (const.term=1.) */
   e = exp(-alpha / p[2]);
   c = 1 + p[1] * e + p[3] * alpha;
   /* derivatives */
   dcdp[1] = e;
   dcdp[2] = p[1] * e * alpha / (p[2] * p[2]);
   dcdp[3] = alpha;

   Scale = c;
}
