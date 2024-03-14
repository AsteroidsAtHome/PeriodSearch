#pragma once

#define POINTS_MAX         2047             /* max number of data points in one lc. 2000 */
#define MAX_N_OBS         10000             /* max number of data points 20000 */
#define MAX_LC               23             /* max number of lightcurves 80 */
#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
#define MAX_N_FAC           319             /* max number of facets 383 */
#define MAX_N_ITER           60             /* maximum number of iterations 128, 60*/
#define MAX_N_PAR            87             /* maximum number of parameters 207*/
#define MAX_LM               15             /* maximum degree and order of sph. harm. */
#define N_PHOT_PAR            5             /* maximum number of parameters in scattering law */
#define TINY                  1e-8          /* precision parameter for mu, mu0 */
#define N_POLES              10             /* number of initial poles */

#define PI                 M_PI             /* 3.14159265358979323846 */
#define AU            149597870.691         /* Astronomical Unit [km] */
#define C_SPEED       299792458             /* speed of light [m/s] */

#define DEG2RAD      (PI / 180)
#define RAD2DEG      (180 / PI)

//MUST BE 128 or 64
#define CUDA_BLOCK_DIM 32
#define N80  64

//#define POINTS_MAX         2000             /* max number of data points in one lc. */
//#define MAX_N_OBS         20000             /* max number of data points */
//#define MAX_LC              200             /* max number of lightcurves */
//#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
//#define MAX_N_FAC          1000             /* max number of facets */
//#define MAX_N_ITER          100             /* maximum number of iterations */
//#define MAX_N_PAR           200             /* maximum number of parameters */
//#define MAX_LM               10             /* maximum degree and order of sph. harm. */
//#define N_PHOT_PAR            5             /* maximum number of parameters in scattering  law */
//#define TINY                  1e-8          /* precision parameter for mu, mu0*/
//#define N_POLES              10             /* number of initial poles */

//#define PI                 M_PI             /* 3.14159265358979323846 */
//#define AU            149597870.691         /* Astronomical Unit [km] */
//#define C_SPEED       299792458             /* speed of light [m/s]*/

//#define DEG2RAD      (PI / 180)
//#define RAD2DEG      (180 / PI)

//MUST BE 128 or 64
//#define CUDA_BLOCK_DIM 128