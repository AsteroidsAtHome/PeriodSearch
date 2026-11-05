// ReSharper disable IdentifierTypo
#pragma once

#if defined _WIN32
#include "Windows.h"
#endif

#include <string>
#include <vector>
#include "Enums.h"
#include "arrayHelpers.hpp"

//void trifac(int nrows, int** ifp);
void trifac(const int nrows, std::vector<std::vector<int>>& ifp);

//void areanorm(double t[], double f[], int ndir, int nfac, int** ifp, double at[], double af[], globals &gl);
void areanorm(const std::vector<double>& t, const std::vector<double>& f, int ndir, int nfac,
	const std::vector<std::vector<int>>& ifp, std::vector<double>& at, std::vector<double>& af, globals &gl);

//void sphfunc(int ndir, double at[], double af[]);
void sphfunc(const int ndir, const std::vector<double>& at, const std::vector<double>& af);

//void ellfit(double r[], double a, double b, double c, int ndir, int ncoef, double at[], double af[]);
void ellfit(std::vector<double>& cg, double a, double b, double c, int ndir, int ncoef, const std::vector<double>& at, const std::vector<double>& af);

//void lubksb(double** a, int n, int indx[], double b[]);
void lubksb(const std::vector<std::vector<double>>& a, const int n, const std::vector<int>& indx, std::vector<double>& b);

//void ludcmp(double** a, int n, int indx[], double d[]);
void ludcmp(std::vector<std::vector<double>>& a, const int n, std::vector<int>& indx, std::vector<double>& d);

//int mrqmin(double** x1, double** x2, double x3[], double y[],
//	double sig[], double a[], int ia[], int ma,
//	double** covar, double** alpha, globals &gl);

int mrqmin(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
	std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, const int ma, globals& gl);

//int mrqmin(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
//	std::vector<double>& sig, double a[], int ia[], int ma,
//	double** covar, double** alpha, globals& gl);

void blmatrix(const double bet, const double lam, struct globals& gl);

void covsrt(double** covar, int ma, int ia[], int mfit);

void phasec(double dcdp[], double alpha, double p[], struct globals& gl);

void matrix(double omg, double t, double tmat[][4], double dtm[][4][4], const struct globals& gl);

//double* vector_double(int length);
//int* vector_int(int length);
//double** matrix_double(int rows, int columns);
//double** aligned_matrix_double(int rows, int columns);
//int** matrix_int(int rows, int columns);
//double*** matrix_3_double(int n_1, int n_2, int n_3);
//void deallocate_vector(void* p_x);
//void deallocate_matrix_double(double** p_x, int rows);
//void aligned_deallocate_matrix_double(double** p_x, int rows);
//void deallocate_matrix_int(int** p_x, int rows);

//double dot_product(double a[4], double b[4]);
//void dot_product_new(double a[4], double b[4], double& c);

#if !defined __GNUC__ && defined _WIN32
	bool GetVersionInfo(LPCTSTR filename, int& major, int& minor, int& build, int& revision);
#elif defined __GNUC__
	bool GetVersionInfo(int& major, int& minor, int& build, int& revision);
#endif
std::string GetCompilerInfo();

std::string GetCpuInfo();
SIMDEnum GetBestSupportedSIMD();
void GetSupportedSIMDs();
SIMDEnum CheckSupportedSIMDs(SIMDEnum simd);
std::shared_ptr<CalcStrategy> CreateCalcStrategy(SIMDEnum useOptimization);
const std::string getSIMDEnumName(SIMDEnum simdEnum);
//void prepareLcData(struct globals &gl, const char *filename);
