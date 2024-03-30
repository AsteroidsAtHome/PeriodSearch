#pragma once

#if defined _WIN32
#include "Windows.h"
#endif

#include <string>
#include "Enums.h"

void trifac(int nrows, int** ifp);
void areanorm(double t[], double f[], int ndir, int nfac, int** ifp,
	double at[], double af[]);
void sphfunc(int ndir, double at[], double af[]);
void ellfit(double r[], double a, double b, double c,
	int ndir, int ncoef, double at[], double af[]);
void lubksb(double** a, int n, int indx[], double b[]);
void ludcmp(double** a, int n, int indx[], double d[]);

int mrqmin(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** covar, double** alpha);

void blmatrix(double bet, double lam);
void covsrt(double** covar, int ma, int ia[], int mfit);
void phasec(double dcdp[], double alpha, double p[]);
void matrix(double omg, double t, double tmat[][4], double dtm[][4][4]);
double* vector_double(int length);
int* vector_int(int length);
double** matrix_double(int rows, int columns);
double** aligned_matrix_double(int rows, int columns);
int** matrix_int(int rows, int columns);
double*** matrix_3_double(int n_1, int n_2, int n_3);
void deallocate_vector(void* p_x);
void deallocate_matrix_double(double** p_x, int rows);
void aligned_deallocate_matrix_double(double** p_x, int rows);
void deallocate_matrix_int(int** p_x, int rows);
double dot_product(double a[4], double b[4]);
void dot_product_new(double a[4], double b[4], double& c);

#if !defined __GNUC__ && defined _WIN32
	bool GetVersionInfo(LPCTSTR filename, int& major, int& minor, int& build, int& revision);
#elif defined __GNUC__
	bool GetVersionInfo(int& major, int& minor, int& build, int& revision);
#endif

std::string GetCpuInfo();
SIMDEnum GetBestSupportedSIMD();
void GetSupportedSIMDs();
SIMDEnum CheckSupportedSIMDs(SIMDEnum simd);
void SetOptimizationStrategy(SIMDEnum useOptimization);
const std::string getSIMDEnumName(SIMDEnum simdEnum);
