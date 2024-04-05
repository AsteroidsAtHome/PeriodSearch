/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include "constants.h"

double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
{
   int ncoef0, i, j, k,
       incl[MAX_N_FAC+1];
 
   double cos_alpha, br, cl, cls, alpha, sum, sum0, dnom,
          e[4], e0[4],
          php[N_PHOT_PAR+1], dphp[N_PHOT_PAR+1],
	  mu[MAX_N_FAC+1], mu0[MAX_N_FAC+1+1], s[MAX_N_FAC+1], dbr[MAX_N_FAC+1],
          dsmu[MAX_N_FAC+1], dsmu0[MAX_N_FAC+1],
	  de[4][4], de0[4][4], tmat[4][4],
	  dtm[4][4][4];
   
   ncoef0 = ncoef - 2 - Nphpar;
   cl = exp(cg[ncoef-1]); /* Lambert */
   cls = cg[ncoef];       /* Lommel-Seeliger */
   cos_alpha = dot_product(ee, ee0);
   alpha = acos(cos_alpha);
   for (i = 1; i <= Nphpar; i++)
      php[i] = cg[ncoef0+i];

   phasec(dphp,alpha,php); /* computes also Scale */

   matrix(cg[ncoef0],t,tmat,dtm);

   br = 0;
   /* Directions (and ders.) in the rotating system */
   for (i = 1; i <= 3; i++)
   {
      e[i] = 0;
      e0[i] = 0;
      for (j = 1; j <= 3; j++)
      {
         e[i] = e[i] + tmat[i][j] * ee[j];
         e0[i] = e0[i] + tmat[i][j] * ee0[j];
         de[i][j] = 0;
         de0[i][j] = 0;
         for (k = 1; k <= 3; k++)
	 {
            de[i][j] = de[i][j] + dtm[j][i][k] * ee[k];
            de0[i][j] = de0[i][j] + dtm[j][i][k] * ee0[k];
         }
      }
   } 

   /*Integrated brightness (phase coeff. used later) */
   for (i = 1; i <= Numfac; i++)
   {
      incl[i] = 0;
      mu[i] = e[1] * Nor[i][1] + e[2] * Nor[i][2] + e[3] * Nor[i][3];
      mu0[i] = e0[1] * Nor[i][1] + e0[2] * Nor[i][2] + e0[3] * Nor[i][3];
      if((mu[i] > TINY) && (mu0[i] > TINY)) 
      {
         incl[i] = 1;
         dnom = mu[i] + mu0[i];
         s[i] = mu[i] * mu0[i] * (cl + cls / dnom);
         br = br + Area[i] * s[i];
         dsmu[i] = cls * pow(mu0[i] / dnom,2) + cl * mu0[i];
         dsmu0[i] = cls * pow(mu[i] / dnom,2) + cl * mu[i];
         dbr[i] = Darea[i] * s[i];
      }
    }

   /* Derivatives of brightness w.r.t. g-coeffs */
   for (i = 1; i <= ncoef0-3; i++)
   {
      dyda[i] = 0;
      for (j = 1; j <= Numfac; j++)
         if (incl[j] == 1) 
            dyda[i] = dyda[i] + dbr[j] * Dg[j][i];
      dyda[i] = Scale * dyda[i];
   }
   
   /* Ders. of brightness w.r.t. rotation parameters */
   for (k = 1; k <= 3; k++)
   {
      dyda[ncoef0-3+k] = 0;
      for (i = 1; i <= Numfac; i++)
         if (incl[i] == 1)
	 {
            sum = 0;
            sum0 = 0;
            for (j = 1; j <= 3; j++)
	    {
               sum = sum + Nor[i][j] * de[j][k];
               sum0 = sum0 + Nor[i][j] * de0[j][k];
            }
            dyda[ncoef0-3+k] = dyda[ncoef0-3+k] + Area[i] * (dsmu[i] * sum + dsmu0[i] * sum0);
          }
      dyda[ncoef0-3+k] = Scale * dyda[ncoef0-3+k];
   }
   
   /* Ders. of br. w.r.t. phase function params. */
   for(i = 1; i <= Nphpar; i++)
      dyda[ncoef0+i] = br * dphp[i];

   /* Ders. of br. w.r.t. cl, cls */
   dyda[ncoef-1] = 0;
   dyda[ncoef] = 0;
   for (i = 1; i <= Numfac; i++)
      if (incl[i] == 1)
      {
         dyda[ncoef-1] = dyda[ncoef-1] + mu[i] * mu0[i] * Area[i];
         dyda[ncoef] = dyda[ncoef] + Area[i] * mu[i] * mu0[i] / (mu[i] + mu0[i]);
      }
   dyda[ncoef-1] = Scale * dyda[ncoef-1] * cl;
   dyda[ncoef] = Scale * dyda[ncoef];

   /* Scaled brightness */
   br *= Scale;
   
   return(br);
}
