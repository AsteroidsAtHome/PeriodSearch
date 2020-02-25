/* dot product of two vectors
   input: vectors a, b
   output: dot product a.b

   8.11.2006
*/

#include "stdafx.h"

double host_dot_product(double a[4], double b[4])
{

  double c;

  c = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

  return(c);

}


