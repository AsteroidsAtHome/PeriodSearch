/* Form the vertex triplets of standard triangulation facets 
   converted from Mikko's fortran code

   8.11.2006
*/

#include "globals.h"
#include "declarations.h"

void trifac(int nrows, int **ifp)
{
   int nnod, i, j, j0, j1, j2, j3, ntri;
   
   int **nod;

   nod = matrix_int(2*nrows, 4*nrows);

   nnod = 1; /* index from 1 to number of vertices */
   nod[0][0] = nnod;
   for (i = 1; i <= nrows; i++)
      for (j = 0; j <= 4 * i - 1; j++)
      {
         nnod++;
         nod[i][j] = nnod;
         if (j == 0) nod[i][4*i] = nnod;
      }
   for (i = nrows - 1; i >= 1; i--)
      for (j = 0; j <= 4 * i - 1; j++)
      {
         nnod++;
         nod[2*nrows-i][j] = nnod;
         if (j == 0) nod[2*nrows-i][4*i] = nnod;
       }

   nod[2*nrows][0] = nnod + 1;
   ntri = 0;

   for (j1 = 1; j1 <= nrows; j1++)
      for (j3 = 1; j3 <= 4; j3++)
      {
         j0 = (j3-1) * j1;
         ntri++;
         ifp[ntri][1] = nod[j1-1][j0-(j3-1)];
         ifp[ntri][2] = nod[j1][j0];
         ifp[ntri][3] = nod[j1][j0+1];                            
         for (j2 = j0 + 1; j2 <= j0 + j1 - 1; j2++)
	 {
            ntri++;
            ifp[ntri][1] = nod[j1][j2];
            ifp[ntri][2] = nod[j1-1][j2-(j3-1)];
            ifp[ntri][3] = nod[j1-1][j2-1-(j3-1)];
            ntri++;
            ifp[ntri][1] = nod[j1-1][j2-(j3-1)];
            ifp[ntri][2] = nod[j1][j2];
            ifp[ntri][3] = nod[j1][j2+1];
          }
       }

   /* Do the lower hemisphere */
   for (j1 = nrows + 1; j1 <= 2 * nrows; j1++)
      for (j3 = 1; j3 <= 4; j3++)
      {
         j0 = (j3 - 1) * (2 * nrows - j1);
         ntri++;
         ifp[ntri][1] = nod[j1][j0];
         ifp[ntri][2] = nod[j1-1][j0+1+(j3-1)];
         ifp[ntri][3] = nod[j1-1][j0+(j3-1)];               
         for (j2 = j0 + 1; j2 <= j0 + (2 * nrows - j1); j2++)
	 {
            ntri++;
            ifp[ntri][1] = nod[j1][j2];
            ifp[ntri][2] = nod[j1-1][j2+(j3-1)];
            ifp[ntri][3] = nod[j1][j2-1];
            ntri++;
            ifp[ntri][1] = nod[j1][j2];
            ifp[ntri][2] = nod[j1-1][j2+1+(j3-1)];
            ifp[ntri][3] = nod[j1-1][j2+(j3-1)];
         }
      }


   // NOTE: For unit tests reference only
   /*for (int ik = 1; ik <= 3; ik++) {
       printf("\nifp[x][%d]:\n", ik);
       for (int ii = 1; ii < ntri; ii++)
       {
           printf("%d, ", ifp[ii][ik]);
           if (ii % 20 == 0)
               printf("\n");
       }
   }*/

   deallocate_matrix_int(nod, 2*nrows);      
}
