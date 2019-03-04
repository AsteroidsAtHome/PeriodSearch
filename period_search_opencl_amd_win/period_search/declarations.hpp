#pragma once
#include <CL/cl.h>
#include "AnglesOfNormals.hpp"
#include "LightPoint.h"
#include "Coordinates.hpp"

void trifac(const int &nrows, int **ifp);
void areanorm(double *theta, double *phi, const int &ndir, int **ifp, struct AnglesOfNormals& normals);
void sphfunc(const struct AnglesOfNormals &normals);
void ellfit(double r[], double a, double b, double c, const int& ncoef, const struct AnglesOfNormals& normals);

void lubksb(double **a, int n, const int indx[], double b[]);
void ludcmp(double **a, int n, int indx[]);
int mrqmin(const CoordinatesDouble3& ndv, const LightPoints<double>& points, //double x3[], double y[],
            double sig[], std::vector<double> &a, const std::vector<int> &ia, const int& ma,
        double **covar, double **alpha, const int &mfit, struct MrqminContext &ctx);
double mrqcof(const CoordinatesDouble3& ndv, const LightPoints<double>& lightPoints, //double x3[], double y[],
              double sig[], std::vector<double> &a, const std::vector<int> &ia, int ma,
            double **alpha, std::vector<double> &beta, const int &mfit);
//double mrqcof(double **x1, double **x2, double x3[], double y[],
//    double sig[], double a[], int ia[], int ma,
//    double **alpha, double beta[], int mfit, int lastone, int lastma,
//    cl_double _Fc[], cl_double _Fs[], cl_double _Dsph[], cl_double _Dg[]);
void curv(const std::vector<double> &cg);
void blmatrix(const double& beta, const double& lambda);
double conv(const int& nc, const int& ma, double *dres);
//void gauss_1(double **aa, int n, double b[]);
void covsrt(double **covar, int ma, int ia[], int mfit);
void phasec(double dcdp[], double alpha, double p[]);
void matrix(const double& omg, const double& t, double tmat[][4], double dtm[][4][4]);
double Bright2(const PairDouble3 &ndv, const double &t, const std::vector<double> &cg, double *dyda, int ncoef);
//void shell(int n, double a[], int index[]);

double *vector_double(int length);
int *vector_int(int length);
double **matrix_double(int rows, int columns);
int **matrix_int(int rows, int columns);
double ***matrix_3_double(int n_1, int n_2, int n_3);
void deallocate_vector(void *p_x);
void deallocate_matrix_double(double **p_x, int rows);
void deallocate_matrix_int(int **p_x, int rows);

//void deallocate_matrix_3(void ***p_x, int n_1, int n_2);

double host_dot_product(double a[], double b[]);
/*
double hapke(double mi0, double mi, double alfa, double sc_param[]);

void sph2cart(double *vektor);
void rotation(double vector[], char *axis, double angle, char *direction);
void inverze(double **a, int n);
void cross_product(double a[], double b[], double c[]);
double norm(double a[]);
double ran1(long *idum);
double gasdev(long *idum);

double raytracing(double sl[], double poz[], int n_fac,
                  double **d, double **e, double **f, double **o,
          double **normal, double ds[],
              int n_over_horiz[], int **fac_list, char *sc_law,
          double sc_param[]);
void precomp(int n_fac, double *x, double *y, double *z, int **fac,
             double **d, double **e, double **f, double **o,
         double **normal, double *ds,
         int *n_over_horiz, int **fac_list);

void matrix_YORP(double omg, double yorp, double t, double tmat[][4], double dtm[][4][4]);
double bright_YORP(double ee[], double ee0[], double t, double cg[],
            double dyda[], int ncoef);

int mrqmin_ell(double **x1, double **x2, double x3[], double y[],
            double sig[], double a[], int ia[], int ma,
        double **covar, double **alpha, double (*funcs)());
double mrqcof_ell(double **x1, double **x2, double x3[], double y[],
              double sig[], double a[], int ia[], int ma,
          double **alpha, double beta[], double (*funcs)());
*/
int gauss_errc(double **aa, int n, std::vector<double> &b);
/*
void matrix_ell(double omg, double fi0, double t, double tmat[][4], double dtm[][4][4]);
double bright_ell(double ee[], double ee0[], double t, double cg[],
            double dyda[], int ncoef);

double bright_ell_YORP(double ee[], double ee0[], double t, double cg[],
            double dyda[], int ncoef);
void matrix_ell_YORP(double omg, double fi0, double yorp, double t, double tmat[][4], double dtm[][4][4]);
*/
void curv1D(double cg[]);